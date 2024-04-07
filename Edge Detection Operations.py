import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Edge Detection - Roberts
roberts_kernel_x = np.array([[1, 0], [0, -1]])
roberts_kernel_y = np.array([[0, 1], [-1, 0]])
edges_roberts_x = cv2.filter2D(image, -1, roberts_kernel_x)
edges_roberts_y = cv2.filter2D(image, -1, roberts_kernel_y)
edges_roberts = cv2.magnitude(edges_roberts_x, edges_roberts_y)

# Edge Detection - Prewitt
prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
edges_prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x)
edges_prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y)
edges_prewitt = cv2.magnitude(edges_prewitt_x, edges_prewitt_y)

# Edge Detection - Sobel
edges_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
edges_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(edges_sobel_x, edges_sobel_y)

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1), plt.imshow(edges_roberts, cmap='gray')
plt.title('Roberts Edge Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(edges_prewitt, cmap='gray')
plt.title('Prewitt Edge Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(edges_sobel, cmap='gray')
plt.title('Sobel Edge Detection'), plt.xticks([]), plt.yticks([])

plt.show()
