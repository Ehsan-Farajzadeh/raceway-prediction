import cv2
import numpy as np
import os
import csv

def measure_lengths(image):
    """
    Measures raceway height, depth up, and depth down from a given image.

    Parameters:
        image (ndarray): Input thermal image.

    Returns:
        tuple: (height_length, depth_up_length, depth_down_length)
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary threshold to isolate the raceway region (assumed purple in grayscale)
    _, purple_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Set horizontal range for vertical height measurement
    height_start_x = 417
    height_end_x = 923
    height_lengths = []

    for col in range(height_start_x, height_end_x):
        for row in range(image.shape[0]):
            if purple_mask[row, col] == 0:
                height_lengths.append(row)
                break

    # Define starting points for depth measurements
    depth_up_start = (418, 531)
    depth_down_start = (415, 595)

    # Detect lines in the mask
    lines = cv2.HoughLines(purple_mask, 1, np.pi / 180, threshold=100)
    depth_up_length = 0
    depth_down_length = 0

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho

            # Depth up
            dx_up = int(x0 + depth_up_start[0] - depth_up_start[1] * b)
            dy_up = int(y0 + depth_up_start[1] + depth_up_start[0] * a)
            if 0 <= dx_up < image.shape[1] and 0 <= dy_up < image.shape[0]:
                if purple_mask[dy_up, dx_up] > 0:
                    depth_up_length += 1

            # Depth down
            dx_down = int(x0 + depth_down_start[0] - depth_down_start[1] * b)
            dy_down = int(y0 + depth_down_start[1] + depth_down_start[0] * a)
            if 0 <= dx_down < image.shape[1] and 0 <= dy_down < image.shape[0]:
                if purple_mask[dy_down, dx_down] > 0:
                    depth_down_length += 1

    height_length = max(height_lengths) if height_lengths else 0

    return height_length, depth_up_length, depth_down_length


def process_images(folder_path="raceway_images", output_csv="measurements.csv"):
    """
    Processes thermal images from CFD output and saves raceway size measurements to a CSV.

    Parameters:
        folder_path (str): Path to the folder containing thermal images.
        output_csv (str): Path to the CSV file to save measurements.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Injection Rate", "Time", "Height", "Depth Up", "Depth Down"])

        for rate in [100, 120, 140, 150, 160, 180, 200]:
            for time in np.arange(0.1, 3.6, 0.1):
                filename = os.path.join(folder_path, f"COG_{rate}_{time:.1f}.jpg")

                if os.path.exists(filename):
                    image = cv2.imread(filename)
                    h, du, dd = measure_lengths(image)
                    writer.writerow([rate, time, h, du, dd])
                    print(f"[✓] Rate: {rate}, Time: {time:.1f}s → Height: {h}, Depth Up: {du}, Depth Down: {dd}")
                else:
                    print(f"[!] Image not found: {filename}")

    print(f"\nAll measurements saved to '{output_csv}'")


if __name__ == "__main__":
    process_images()

