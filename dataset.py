import pandas as pd
import numpy as np
import os
from PIL import Image
import hashlib
import random
from collections import defaultdict
import albumentations as A

TEST_DIR = "test"
TRAINING_DIR = "training"
FINAL_TRAIN_CSV_PATH = "train_final.csv"


def main():
    training_labels_path = os.path.join(TRAINING_DIR, "labels.csv")
    train_df = pd.read_csv(training_labels_path, index_col="ImageId")

    all_exist = True
    for i in range(1, train_df.shape[0] + 1):
        if not os.path.exists(os.path.join(TRAINING_DIR, f"{i}.png")):
            all_exist = False
            print(f"Found missing image: {i}.png")
    print("All images found!" if all_exist else "Some images not found!")

    for file in os.listdir(TRAINING_DIR):
        if file.endswith(".png"):
            img_path = os.path.join(TRAINING_DIR, file)
            with Image.open(img_path) as img:
                width, height = img.size
                assert width == 96 and height == 96
    print("Training images 96x96 size validated")

    for file in os.listdir(TEST_DIR):
        if file.endswith(".png"):
            img_path = os.path.join(TRAINING_DIR, file)
            with Image.open(img_path) as img:
                width, height = img.size
                assert width == 96 and height == 96
    print("Test images 96x96 size validated")

    hashes = defaultdict(list)

    for filename in os.listdir(TRAINING_DIR):
        if filename.endswith(".png"):
            path = os.path.join(TRAINING_DIR, filename)
            with open(path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            hashes[file_hash].append(filename)

    duplicates = [files for files in hashes.values() if len(files) > 1]
    print(f"Found {len(duplicates)} duplicate instances in training dataset!")
    total_removed = 0

    train_nodup_df = train_df
    for dup in duplicates:
        orig = dup[0]
        same = dup[1:]
        same_ids = list(map(lambda x: int(x[:-4]), same))
        train_nodup_df = train_nodup_df.drop(same_ids)
        total_removed += len(same)
    print("Total number of removed images (training): ", total_removed)

    hashes = defaultdict(list)
    for filename in os.listdir(TEST_DIR):
        if filename.endswith(".png"):
            path = os.path.join(TEST_DIR, filename)
            with open(path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            hashes[file_hash].append(filename)

    duplicates = [files for files in hashes.values() if len(files) > 1]
    print(f"Found {len(duplicates)} duplicate instances in test dataset!")

    for dup in duplicates:
        same = dup[1:]
        for file in same:
            file_path = os.path.join(TEST_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                total_removed += 1
    print("Total number of removed images (test + training): ", total_removed)

    hashes = defaultdict(list)

    for filename in os.listdir(TEST_DIR):
        if filename.endswith(".png"):
            path = os.path.join(TEST_DIR, filename)
            with open(path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            hashes[file_hash].append(path)

    for image_id in train_nodup_df.index:
        filename = str(image_id) + ".png"
        path = os.path.join(TRAINING_DIR, filename)
        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        hashes[file_hash].append(path)

    duplicates = [files for files in hashes.values() if len(files) > 1]
    print(
        f"Found {len(duplicates)} duplicate instances between test and training dataset!"
    )

    to_drop = []
    for d in duplicates:
        dup = d[0] if TRAINING_DIR in d[0] else d[1]
        _, image_name = dup.split("\\")
        image_id = int(image_name[:-4])
        to_drop.append(image_id)
        total_removed += 1
    train_nodup_df = train_nodup_df.drop(index=to_drop)
    print("Total number of removed images (test + training + intra): ", total_removed)

    print(f"Total number of training images (no duplicates): {train_nodup_df.shape[0]}")

    SEED = 41

    random.seed(SEED)
    np.random.seed(SEED)

    aug_folder = "augmented/"
    os.makedirs(aug_folder, exist_ok=True)

    keypoint_cols = [
        "left_eye_center",
        "right_eye_center",
        "left_eye_inner_corner",
        "left_eye_outer_corner",
        "right_eye_inner_corner",
        "right_eye_outer_corner",
        "left_eyebrow_inner_end",
        "left_eyebrow_outer_end",
        "right_eyebrow_inner_end",
        "right_eyebrow_outer_end",
        "nose_tip",
        "mouth_left_corner",
        "mouth_right_corner",
        "mouth_center_top_lip",
        "mouth_center_bottom_lip",
    ]

    def extract_keypoints(row):
        kpts = []
        for feat in keypoint_cols:
            x = row[f"{feat}_x"]
            y = row[f"{feat}_y"]
            kpts.append([x, y])
        return kpts

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=85, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        seed=SEED,
    )

    aug_rows = []

    for img_id, row in train_nodup_df.iterrows():
        img_path = os.path.join(TRAINING_DIR, f"{img_id}.png")
        image = np.array(Image.open(img_path).convert("RGB"))

        keypoints = extract_keypoints(row)

        augmented = transform(image=image, keypoints=keypoints)
        img_aug = Image.fromarray(augmented["image"])
        kpts_aug = augmented["keypoints"]

        new_img_name = f"aug_{img_id}.png"
        save_path = os.path.join(aug_folder, new_img_name)
        img_aug.save(save_path)

        kpt_dict = {}
        for feat, (x, y) in zip(keypoint_cols, kpts_aug):
            kpt_dict[f"{feat}_x"] = x
            kpt_dict[f"{feat}_y"] = y
        kpt_dict["ImageId"] = new_img_name[:-4]
        aug_rows.append(kpt_dict)

    df_aug = pd.DataFrame(aug_rows)
    df_aug.set_index("ImageId", inplace=True)
    print(f"Generated total of {df_aug.shape[0]} augmented images!")

    train_final_df = pd.concat([train_nodup_df, df_aug], axis=0)
    print(
        f"Merged original images with augmented: {train_final_df.shape[0]} instances!"
    )
    train_final_df = train_final_df.fillna(0.0)
    print(f"Filled NAs with zeroes")
    train_final_df.to_csv(FINAL_TRAIN_CSV_PATH)


if __name__ == "__main__":
    main()
