# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import subprocess
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
from numpy import linalg
import PIL.Image
import torch

import legacy
import torch.nn.functional as F
import scipy.interpolate
import imageio
from tqdm import tqdm
from typing import List, Optional
import moviepy.editor


# ---------------------------------------------------------------------------


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def size_range(s: str) -> List[int]:
    '''Accept a range 'a-c' and return as a list of 2 ints.'''
    return [int(v) for v in s.split('-')][::-1]


def get_w_from_seed(G, batch_sz, device, truncation_psi=1.0, seed=None, centroids_path=None, class_idx=None, conditional_truncation=False):
    """Get the dlatent from a list of random seeds, using the truncation trick (this could be optional)"""

    if G.c_dim != 0:
        # sample random labels if no class idx is given
        if class_idx is None:
            class_indices = np.random.RandomState(seed).randint(low=0, high=G.c_dim, size=(batch_sz))
            class_indices = torch.from_numpy(class_indices).to(device)
            w_avg = G.mapping.w_avg.index_select(0, class_indices)
        else:
            w_avg = G.mapping.w_avg[class_idx].unsqueeze(0).repeat(batch_sz, 1)
            class_indices = torch.full((batch_sz,), class_idx).to(device)

        labels = F.one_hot(class_indices, G.c_dim)

    else:
        w_avg = G.mapping.w_avg.unsqueeze(0)
        labels = None
        if class_idx is not None:
            print('Warning: --class is ignored when running an unconditional network')

    z = np.random.RandomState(seed).randn(batch_sz, G.z_dim)
    z = torch.from_numpy(z).to(device)
    w = G.mapping(z, labels, conditional_truncation=conditional_truncation)

    # multimodal truncation
    if centroids_path is not None:

        with dnnlib.util.open_url(centroids_path, verbose=False) as f:
            w_centroids = np.load(f)
        w_centroids = torch.from_numpy(w_centroids).to(device)
        w_centroids = w_centroids[None].repeat(batch_sz, 1, 1)

        # measure distances
        dist = torch.norm(w_centroids - w[:, :1], dim=2, p=2)
        w_avg = w_centroids[0].index_select(0, dist.argmin(1))

    w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
    w = w_avg + (w - w_avg) * truncation_psi


    return w


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def gen_interp_video(G, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, truncation_psi=1, noise_mode='const', conditional_truncation=False, device=torch.device('cuda'), centroids_path=None, class_idx=None, **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    if class_idx is None:
        class_idx = [None] * len(seeds)
    elif len(class_idx) == 1:
        class_idx = [class_idx] * len(seeds)
    assert len(all_seeds) == len(class_idx), "Seeds and class-idx should have the same length"

    ws = []
    for seed, cls in zip(all_seeds, class_idx):
        ws.append(
            get_w_from_seed(G, 1, device, truncation_psi, seed=seed,
                                      centroids_path=centroids_path, class_idx=cls, conditional_truncation=conditional_truncation)
        )
    ws = torch.cat(ws)

    _ = G.synthesis(ws[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                img = G.synthesis(ws=w.unsqueeze(0), noise_mode=noise_mode)[0]
                imgs.append(img)
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()




#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=int, help='Seed')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--frames', type=int, help='how many frames to produce (with seeds this is frames between each step, with loops this is total length)', default=240, show_default=True)
@click.option('--fps', type=int, help='framerate for video', default=24, show_default=True)
@click.option('--audio_path', type=int, help='audio path for song', default=None, show_default=True)

@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--process', type=click.Choice(['image', 'interpolation','truncation','interpolation-truncation']), default='image', help='generation method', required=True)
@click.option('--scale-type',
                type=click.Choice(['pad', 'padside', 'symm','symmside']),
                default='pad', help='scaling method for --size', required=False)
@click.option('--size', type=size_range, help='size of output (in format x-y)')

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    scale_type: Optional[str],
    size: Optional[List[int]],
    seed: int,
    fps: Optional[int],
    frames: Optional[int],
    truncation_psi: float,
    noise_mode: str,
    category_list: List[int],
    audio_path: Optional[str],
    outdir: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    
    # custom size code from https://github.com/eps696/stylegan2ada/blob/master/src/_genSGAN2.py
    if(size): 
        print('render custom size: ',size)
        print('padding method:', scale_type )
        custom = True
    else:
        custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = size 
    G_kwargs.scale_type = scale_type


    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        # G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    seeds = [seed]*len(category_list)
    output = f'/{outdir}/seed{seed:04d}_video.mp4'

    gen_interp_video(G=G, mp4=output, bitrate='15M', seeds=seeds, shuffle_seed=False, w_frames=30, class_idx=category_list)

    if audio_path:
        mp4_filename = f'/{outdir}/combined.mp4'
        video_clip = moviepy.editor.VideoFileClip(output)
        audio_clip_i = moviepy.editor.AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip_i)
        video_clip.write_videofile(mp4_filename, fps=fps, codec='libx264', audio_codec='aac', bitrate='15M')


    

    


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
