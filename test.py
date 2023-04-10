import json

def main():
    file = open("dataset_copy.json")
    labels = json.load(file)
    files = labels["labels"]

    print(len(files))

main()


