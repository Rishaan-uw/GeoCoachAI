import json
import tarfile

import requests
from huggingface_hub import hf_hub_url


DATASET_ID = "osv5m/osv5m-wds"
SHARD_PATH = "train/0000.tar"
SAMPLE_LIMIT = 5


def main():
    shard_url = hf_hub_url(
        repo_id=DATASET_ID,
        filename=SHARD_PATH,
        repo_type="dataset",
    )

    print("Opening the first OSV5M training shard...")

    with requests.get(
        shard_url,
        stream=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()
        response.raw.decode_content = True

        with tarfile.open(fileobj=response.raw, mode="r|*") as archive:
            samples_found = 0

            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue

                metadata_file = archive.extractfile(member)

                if metadata_file is None:
                    continue

                metadata = json.load(metadata_file)

                print("\nRecord:", member.name)
                print("Available fields:", sorted(metadata.keys()))
                print("Country:", metadata.get("country"))
                print("Latitude:", metadata.get("latitude"))
                print("Longitude:", metadata.get("longitude"))
                print("Sequence:", metadata.get("sequence"))

                samples_found += 1

                if samples_found >= SAMPLE_LIMIT:
                    break

    print("\nInspection complete.")

if __name__ == "__main__":
    main()