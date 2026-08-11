from itertools import islice

from huggingface_hub import HfApi


DATASET_ID = "osv5m/osv5m-wds"


def main():
    api = HfApi()

    print("Connecting to OSV5M...")
    print("First training files:\n")

    files = api.list_repo_tree(
        repo_id=DATASET_ID,
        repo_type="dataset",
        path_in_repo="train",
        recursive=False,
    )

    for entry in islice(files, 10):
        size_mb = getattr(entry, "size", 0) / 1_000_000

        print("Path:", entry.path)
        print(f"Size: {size_mb:.1f} MB")
        print("-" * 50)


if __name__ == "__main__":
    main()