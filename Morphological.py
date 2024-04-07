import cv2
import numpy as np
import matplotlib.pyplot as plt

def perform_morphological_operations(image):
    # Define structuring element
    kernel = np.ones((5,5),np.uint8)
    
    # Erosion
    erosion = cv2.erode(image, kernel, iterations = 1)
    
    # Dilation
    dilation = cv2.dilate(image, kernel, iterations = 1)
    
    # Opening (erosion followed by dilation)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Closing (dilation followed by erosion)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Boundary detection (morphological gradient)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    
    return erosion, dilation, opening, closing, gradient

# Load an image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Perform morphological operations
erosion, dilation, opening, closing, gradient = perform_morphological_operations(image)

# Display original and processed images
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(gradient, cmap='gray')
plt.title('Boundary Detection')
plt.axis('off')

plt.show()
