from PIL import Image
from skimage import filters
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Load your original image using PIL
original_image_path = "E:\Sem 5\Image and Video Processing(CS333)\Project\ProjectImage_2.jpg"
original_pil_image = Image.open(original_image_path).convert("L")  # Convert to grayscale
original_image = np.array(original_pil_image)

otsu_threshold = filters.threshold_otsu(original_image)
otsu_image = original_image > otsu_threshold

# Perform Canny edge detection (for Genetic method)
canny_edges = cv2.Canny(original_image, 50, 150)  # Adjust parameters as needed
genetic_image = canny_edges

Igray = np.array(original_pil_image)

# Define image gradients manually or using Sobel operators
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Gy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])

# Calculate image gradients using convolutions or np.gradient
Ix = convolve2d(Igray, Gx, mode='same', boundary='symm', fillvalue=0)
Iy = convolve2d(Igray, Gy, mode='same', boundary='symm', fillvalue=0)

# Define a simple fuzzy edge detection function
def fuzzy_edge_detection(Ix, Iy, threshold=50):
    Iout = np.sqrt(Ix**2 + Iy**2)  # Combine gradients using magnitude
    # Apply a threshold to determine edges
    Ieval = np.where(Iout > threshold, 1, 0)
    return Ieval

# Apply fuzzy edge detection
Ieval = fuzzy_edge_detection(Ix, Iy, threshold=50)  # Adjust threshold as needed

fuzzy_image = Ieval

# Define functions to calculate MSE, PSNR, and SNR
def calculate_mse(original, processed):
    return np.mean((original - processed) ** 2)

def calculate_psnr(mse):
    max_pixel = 255.0  # Assuming pixel values range from 0 to 255
    return 10 * np.log10((max_pixel ** 2) / mse)

def calculate_snr(original, processed):
    mean_original = np.mean(original)
    sd = np.std(original - processed)
    return 10 * np.log10((mean_original ** 2) / (sd ** 2))

mse_otsu = np.sum(np.square(original_image - otsu_image)) / (original_image.size)
mse_genetic = np.sum(np.square(original_image - genetic_image)) / (original_image.size)
mse_fuzzy = np.sum(np.square(original_image - fuzzy_image)) / (original_image.size)

# Calculate PSNR for each method
psnr_otsu = calculate_psnr(mse_otsu)
psnr_genetic = calculate_psnr(mse_genetic)
psnr_fuzzy = calculate_psnr(mse_fuzzy)

# Calculate SNR for each method
snr_otsu = calculate_snr(original_image, otsu_image)
snr_genetic = calculate_snr(original_image, genetic_image)
snr_fuzzy = calculate_snr(original_image, fuzzy_image)

# Comparison of PSNR, MSE, SNR values
print("Comparison of Metrics:")
print("Method\t\tPSNR (dB)\tMSE\t\tSNR (dB)")
print(f"OTSU\t\t{psnr_otsu:.2f}\t\t{mse_otsu:.2f}\t\t{snr_otsu:.2f}")
print(f"Canny\t\t{psnr_canny:.2f}\t\t{mse_canny:.2f}\t\t{snr_canny:.2f}")
print(f"Fuzzy\t\t{psnr_fuzzy:.2f}\t\t{mse_fuzzy:.2f}\t\t{snr_fuzzy:.2f}")
