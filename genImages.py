import os
import csv
import numpy as np
from PIL import Image

TEST_FILE: str = "test.csv"
TRAINING_FILE: str = "training.csv"

TEST_DIR: str = "test"
TRAINING_DIR: str = "training"


def genImage(nums: list[int], path: str) -> None:
    pixels = np.array(nums, dtype=np.uint8).reshape((96, 96))
    img = Image.fromarray(pixels, mode="L")
    img.save(path)


def genTests() -> None:
    print("[-] Generating Test Images...")
    with open(TEST_FILE, newline="") as f:
        test_reader = csv.reader(f)
        next(test_reader)  # Headers
        for row in test_reader:
            id = row[0]
            nums = list(map(int, row[1].split()))
            path_out = os.path.join(TEST_DIR, f"{id}.png")
            genImage(nums, path_out)
    print("[+] Generated Test Images.")


def genTraining() -> None:
    print("[-] Generating Training Images & Labels...")
    id: int = 1
    data: list[str] = []
    with open(TRAINING_FILE, newline="") as f:
        train_reader = csv.reader(f)
        headers = next(train_reader)
        data.append(["ImageId"] + headers[:-1])
        for row in train_reader:
            data.append([str(id)] + row[:-1])
            nums = list(map(int, row[-1].split()))
            path_out = os.path.join(TRAINING_DIR, f"{id}.png")
            genImage(nums, path_out)
            id += 1

    training_labels = os.path.join(TRAINING_DIR, f"labels.csv")
    with open(training_labels, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print("[+] Generated Training Images & Labels.")


def main() -> None:
    """
    Script generates images from given test and training files
    Labels for training saved along the images in TRAINING_DIR
    """
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    genTests()
    genTraining()


if __name__ == "__main__":
    main()
