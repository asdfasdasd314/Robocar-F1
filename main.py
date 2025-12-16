import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_channel_stdev(img):
    """
    Calculates the standard deviation across the R, G, and B channels
    for every pixel in the input image.

    Args:
        img (np.ndarray): The input image array (assumed to be BGR format by cv2).

    Returns:
        np.ndarray: A 2D array where each element is the standard deviation
                    of the channels for the corresponding pixel.
    """
    # OpenCV loads images as BGR by default, but the math works the same:
    # Channel 0, Channel 1, Channel 2
    c0, c1, c2 = cv2.split(img)

    # 1. Calculate the mean (average) of the three channels for each pixel
    avg = (c0.astype(np.float32) + c1.astype(np.float32) + c2.astype(np.float32)) / 3

    # 2. Calculate the variance: sum of squared differences from the mean, divided by N (which is 3)
    # Variance = [(c0 - avg)^2 + (c1 - avg)^2 + (c2 - avg)^2] / 3
    variance = (
        (c0 - avg) ** 2 +
        (c1 - avg) ** 2 +
        (c2 - avg) ** 2
    ) / 3

    # 3. Calculate the standard deviation (square root of the variance)
    std_dev = np.sqrt(variance)

    mask = (std_dev < 10) & (c0 > 100) & (c1 > 100) & (c2 > 100)

    return mask

# --- Example Usage (Run this locally after loading your image) ---
# Replace 'your_image_path.png' with the actual path to your image file
try:
    image = cv2.imread('red_img.png')
    if image is None:
        print("Error: Could not load image. Check the file path.")
    else:
        stdev_map = calculate_channel_stdev(image)
        import numpy as np
        import matplotlib.pyplot as plt

        # ASSUMPTION: 'stdev_map' is the 2D array of per-pixel standard deviations 
        # that you calculated using the formula we discussed.
        # You need to run the calculation on your machine first and pass the result here.
        # For example, if you save the result of the stdev calculation to a file, load it here.

        # --- Example of plotting assuming stdev_map is already calculated ---

        # Plotting
        plt.figure(figsize=(10, 8))

        # The imshow function maps the standard deviation values to colors
        # 'viridis' is a good color scheme for this type of quantitative data.
        img_plot = plt.imshow(stdev_map, cmap='viridis', interpolation='nearest')

        # Add a colorbar (the scale)
        cbar = plt.colorbar(img_plot, orientation='vertical', shrink=0.75)
        cbar.set_label('Standard Deviation Value (Floating Point)', rotation=270, labelpad=15, fontsize=12)

        plt.title('Per-Pixel Color Channel Standard Deviation Map', fontsize=14)
        plt.xlabel('X-coordinate (Column)', fontsize=12)
        plt.ylabel('Y-coordinate (Row)', fontsize=12)

        # Save the figure
        output_filename = 'stdev_map_visualization.png'
        plt.savefig(output_filename)
        plt.show() # Use plt.show() if running locally
        plt.close()

        print(f"Visualization saved as {output_filename}")
except Exception as e:
    print(f"An error occurred: {e}")