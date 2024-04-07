import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Smoothing - Averaging Filter
averaging_kernel = np.ones((5, 5), np.float32) / 25
smoothed_image_averaging = cv2.filter2D(image, -1, averaging_kernel)

# Smoothing - Median Filter
smoothed_image_median = cv2.medianBlur(image, 5)

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(smoothed_image_averaging, cmap='gray')
plt.title('Averaging Filter'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(smoothed_image_median, cmap='gray')
plt.title('Median Filter'), plt.xticks([]), plt.yticks([])

plt.show()
