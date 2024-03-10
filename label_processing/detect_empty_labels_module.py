# Import third-party libraries
import glob
import os
from PIL import Image
import shutil


def find_empty_labels(folder_path: str, output_folder: str, threshold: float = 0.01, crop_margin: float = 0.1) -> None:
    """
    Find and move empty and non-empty labels to respective folders.

    Args:
        folder_path (str): Path to the directory containing input images.
        output_folder (str): Path to the directory where filtered images will be stored.
        threshold (float): Threshold for classifying empty labels. Defaults to 0.01.
        crop_margin (float): Margin for cropping images. Defaults to 0.1.

    Returns:
        None
    """
    empty_folder = os.path.join(output_folder, "empty")
    not_empty_folder = os.path.join(output_folder, "not_empty")
    os.makedirs(empty_folder, exist_ok=True)
    os.makedirs(not_empty_folder, exist_ok=True)

    for filename in glob.iglob(os.path.join(folder_path, '*')):
        if os.path.isfile(filename):
            try:
                img = Image.open(filename)
                width, height = img.size

                # Crop image
                start_width = int(width * crop_margin)
                end_width = width - start_width
                start_height = int(height * crop_margin)
                end_height = height - start_height

                black_pixels_proportion = detect_dark_pixels(
                    img, start_width, end_width, start_height, end_height
                )

                if black_pixels_proportion < threshold:
                    shutil.move(filename, os.path.join(empty_folder, os.path.basename(filename)))
                else:
                    shutil.move(filename, os.path.join(not_empty_folder, os.path.basename(filename)))

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def detect_dark_pixels(image: Image, start_width: int, end_width: int, start_height: int, end_height: int, threshold: int = 100) -> float:
    """
    Detect the proportion of dark pixels in an image.

    Args:
        image (Image): Input image.
        start_width (int): Starting width for image cropping.
        end_width (int): Ending width for image cropping.
        start_height (int): Starting height for image cropping.
        end_height (int): Ending height for image cropping.
        threshold (int): Threshold for classifying dark pixels. Defaults to 100.

    Returns:
        float: Proportion of dark pixels.
    """
    black_pixels = 0
    total_pixels = 0
    for w in range(start_width, end_width):
        for h in range(start_height, end_height):
            color_tuple = image.getpixel((w, h))
            total_pixels += 1
            # Calculate pixel brightness based on the sum of RGB values
            brightness = sum(color_tuple) / 3
            if brightness < threshold:
                black_pixels += 1
    return black_pixels / total_pixels