import json
import tarfile
from itertools import islice

import requests
from huggingface_hub import HfApi, hf_hub_url


DATASET_ID = "osv5m/osv5m-wds"
TARGET_COUNTRIES = {"AU", "SG", "JP"}

# Start small. We can increase this after confirming it works.
SHARDS_TO_CHECK = 20


def read_first_country(shard_path):
    shard_url = hf_hub_url(
        repo_id=DATASET_ID,
        filename=shard_path,
        repo_type="dataset",
    )

    with requests.get(
        shard_url,
        stream=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()
        response.raw.decode_content = True

        with tarfile.open(fileobj=response.raw, mode="r|*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue

                metadata_file = archive.extractfile(member)

                if metadata_file is None:
                    continue

                metadata = json.load(metadata_file)
                return metadata.get("country")

    return None


def main():
    api = HfApi()

    shard_entries = api.list_repo_tree(
        repo_id=DATASET_ID,
        repo_type="dataset",
        path_in_repo="train",
        recursive=False,
    )

    shard_paths = [
        entry.path
        for entry in shard_entries
        if entry.path.endswith(".tar")
    ]

    shard_paths.sort()

    print(f"Found {len(shard_paths)} training shards.")
    print(f"Inspecting the first {SHARDS_TO_CHECK}...\n")

    matches = []

    for shard_path in islice(shard_paths, SHARDS_TO_CHECK):
        try:
            country = read_first_country(shard_path)
            marker = " <-- target" if country in TARGET_COUNTRIES else ""

            print(f"{shard_path}: {country}{marker}")

            if country in TARGET_COUNTRIES:
                matches.append((shard_path, country))

        except Exception as error:
            print(f"{shard_path}: ERROR - {error}")

    print("\nTarget matches:")

    if matches:
        for shard_path, country in matches:
            print(f"{country}: {shard_path}")
    else:
        print("No target countries found in this group of shards.")


if __name__ == "__main__":
    main()