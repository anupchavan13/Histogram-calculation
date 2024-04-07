# Histogram-calculation

import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist, bins = np.histogram(gray_image.flatten(), 256, [0,256])
    
    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    return equalized_image

# Load an image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path)

# Apply histogram equalization
equalized_image = histogram_equalization(image)

# Display original and equalized images
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.show()
