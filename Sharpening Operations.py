import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Sharpening - Laplacian Filter
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
sharpened_image_laplacian = cv2.filter2D(image, -1, laplacian_kernel)

# Sharpening - High Boost Filtering
A = 1.5  # You can adjust this parameter
smoothed_image_high_boost = cv2.filter2D(image, -1, laplacian_kernel)
sharpened_image_high_boost = cv2.addWeighted(image, 1, smoothed_image_high_boost, A, 0)

# Sharpening - Unsharp Masking
smoothed_image_unsharp = cv2.GaussianBlur(image, (5, 5), 10)
sharpened_image_unsharp = cv2.addWeighted(image, 1.5, smoothed_image_unsharp, -0.5, 0)

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1), plt.imshow(sharpened_image_laplacian, cmap='gray')
plt.title('Laplacian Filter'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(sharpened_image_high_boost, cmap='gray')
plt.title('High Boost Filtering'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(sharpened_image_unsharp, cmap='gray')
plt.title('Unsharp Masking'), plt.xticks([]), plt.yticks([])

plt.show()
