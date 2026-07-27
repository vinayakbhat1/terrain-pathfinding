import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = r"F:\resume project\path project\input\map_image.png"
SAVE_PATH  = r"F:\resume project\path project\input\preprocessed.npy"


def preprocess(image_path):
    """
    Loads and preprocesses a satellite image.
    Steps:
      1. Resize to 800x400 — standardizes input, keeps processing fast
      2. Denoise — removes noise so KMeans clusters more accurately
      3. BGR → HSV — separates color from brightness, robust to shadows
    Returns: bgr_clean, hsv, rgb_clean
    """
    image_bgr      = cv2.imread(image_path)
    image_resized  = cv2.resize(image_bgr, (800, 400))
    image_denoised = cv2.fastNlMeansDenoisingColored(image_resized, None, 10, 10, 7, 21)
    image_hsv      = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2HSV)
    image_rgb      = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2RGB)
    return image_denoised, image_hsv, image_rgb


if __name__ == "__main__":
    bgr_clean, hsv, rgb_clean = preprocess(IMAGE_PATH)

    print(f"Resized Shape : {bgr_clean.shape}")
    print(f"HSV Shape     : {hsv.shape}")

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(rgb_clean);                      axes[0].set_title("Step 1: Resized (800x400)"); axes[0].axis('off')
    axes[1].imshow(rgb_clean);                      axes[1].set_title("Step 2: Denoised");          axes[1].axis('off')
    axes[2].imshow(hsv);                            axes[2].set_title("Step 3: HSV Converted");     axes[2].axis('off')
    plt.suptitle("Phase 2 — Preprocessing", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Save
    np.save(SAVE_PATH, hsv)
    print("preprocessed.npy saved!")