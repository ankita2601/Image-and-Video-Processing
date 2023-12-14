import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Image path
image_path = r'.\input.jpg'
ori_img = cv2.imread('.\input.jpg')
img = Image.open(image_path).convert('L')  # Converting to grayscale
Igray = np.array(img)

# function for convolution
def custom_convolve2d(input_signal, kernel):
    input_rows, input_cols = input_signal.shape
    kernel_rows, kernel_cols = kernel.shape

    # Calculating the output size
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    # Initializing the result matrix
    result = np.zeros((output_rows, output_cols))

    # 2D convolution
    for i in range(output_rows):
        for j in range(output_cols):
            result[i, j] = np.sum(input_signal[i:i+kernel_rows, j:j+kernel_cols] * kernel)

    return result

# Definings image gradients 
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Gy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])

# Calculatimg image gradients using convolutions 
Ix = custom_convolve2d(Igray, Gx)
Iy = custom_convolve2d(Igray, Gy)

# print(Ix)

# simple fuzzy edge detection function
def fuzzy_edge_detection(Ix, Iy, threshold=50):
    Iout = np.sqrt(Ix**2 + Iy**2)  # Combining gradients using magnitude
    # Appling  threshold to determine edges
    Ieval = np.where(Iout > threshold, 1, 0)
    return Ieval

# fuzzy edge detection
Ieval = fuzzy_edge_detection(Ix, Iy, threshold=50)
blurred = cv2.GaussianBlur(Igray, (5, 5), 0)
unsharp_mask = cv2.addWeighted(Igray, 1.5, blurred, -0.5, 0)
sharpened_image = cv2.add(Igray, unsharp_mask)

# Visualization of images
plt.figure(figsize=(12, 12))

plt.subplot(4, 2, 1)
# plt.imshow(img)
plt.imshow(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)) 
plt.title('Original image')

plt.subplot(4, 2, 2)
plt.imshow(Igray, cmap='gray')
plt.title('Igray')

plt.subplot(4, 2, 3)
plt.imshow(Ix, cmap='gray')
plt.title('Ix')

plt.subplot(4, 2, 4)
plt.imshow(Iy, cmap='gray')
plt.title('Iy')

plt.subplot(4, 2, 5)
plt.imshow(Ieval, cmap='gray')
plt.title('Fuzzy Edges')

plt.subplot(4, 2, 6)
plt.imshow(np.invert(Ieval), cmap='gray')
plt.title('Fuzzy Edges 2')
plt.subplot(4, 2, 7)
plt.imshow(Igray, cmap='gray')
plt.title('Igray')

plt.subplot(4, 2, 8)
plt.imshow(sharpened_image, cmap='gray')
plt.title('sharpened_image')



plt.tight_layout()
plt.show()
