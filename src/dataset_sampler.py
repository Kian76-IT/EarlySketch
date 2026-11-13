
import json
import random
import requests
from pathlib import Path

# Configuration
CATEGORIES = [
    "cat",
    "dog",
    "bird",
    "fish",
    "airplane",
    "car",
    "bicycle",
    "house",
    "tree",
    "flower"
]

SAMPLE_SIZE = 500
RANDOM_SEED = 42

BASE_URL = (
    "https://storage.googleapis.com/"
    "quickdraw_dataset/full/raw"
)

OUTPUT_DIR = Path("data/sampled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_sample(category, sample_size=500):
    """
    Download a random sample using reservoir sampling.
    """
    url = f"{BASE_URL}/{category}.ndjson"
    rng = random.Random(RANDOM_SEED)
    samples = []
    print(f"\nProcessing: {category}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        for index, line in enumerate(
            response.iter_lines(decode_unicode=True)
        ):
            if not line:
                continue
            drawing = json.loads(line)
            # Reservoir sampling
            if len(samples) < sample_size:
                samples.append(drawing)
            else:
                position = rng.randint(0, index)
                if position < sample_size:
                    samples[position] = drawing

    # Save sampled dataset
    output_file = OUTPUT_DIR / f"{category}.ndjson"
    with open(output_file, "w", encoding="utf-8") as file:
        for drawing in samples:
            file.write(json.dumps(drawing) + "\n")

    print(f"Category: {category}")
    print(f"Total samples: {len(samples)}")
    print(f"Saved to: {output_file}")

def main():
    for category in CATEGORIES:
        download_sample(
            category=category,
            sample_size=SAMPLE_SIZE
        )
    print("\nAll categories completed!")

if __name__ == "__main__":
    main()
