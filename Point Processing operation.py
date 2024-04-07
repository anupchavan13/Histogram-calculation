import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Example point processing operations
# Adjusting brightness by adding a constant value
brightness_adjusted_image = cv2.add(image, 50)

# Adjusting contrast by scaling pixel values
contrast_adjusted_image = cv2.multiply(image, 1.5)

# Applying gamma correction
gamma = 1.5
gamma_corrected_image = np.uint8(cv2.pow(image / 255.0, gamma) * 255)

# Applying thresholding
_, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Display original and processed images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(brightness_adjusted_image, cmap='gray')
plt.title('Brightness Adjusted')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(contrast_adjusted_image, cmap='gray')
plt.title('Contrast Adjusted')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(gamma_corrected_image, cmap='gray')
plt.title('Gamma Corrected')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded')
plt.axis('off')

plt.show()
