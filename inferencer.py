import argparse
import torch
import cv2
import numpy as np
import random
from train import WiderCNN


def draw_cross(image, x, y, color, size=2, thickness=1):
    """Draw a small cross on the image."""
    x, y = int(x), int(y)
    cv2.line(image, (x - size, y), (x + size, y), color, thickness)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness)


def main():
    parser = argparse.ArgumentParser(description="Torch model inference on an image.")
    parser.add_argument(
        "model_path", type=str, help="Path to the torch model (.pt/.pth)"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "save_path", nargs="?", default=None, help="Optional output image save path"
    )
    args = parser.parse_args()

    state = torch.load(args.model_path, map_location="cpu")
    model = WiderCNN()
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    img = cv2.imread(args.image_path)
    if img is None:
        raise ValueError("Could not read image at provided image_path.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(tensor)

    points = output.reshape(-1, 2).cpu().numpy()

    for x, y in points:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_cross(img, x, y, color)

    if args.save_path:
        cv2.imwrite(args.save_path, img)
        print(f"Saved inference image to {args.save_path}")
    else:
        print("Showing image (not saved). Close window to exit.")
        cv2.imshow("Inference", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
