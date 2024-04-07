import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Roberts edge detection
    roberts_x = cv2.filter2D(gray_image, -1, np.array([[1, 0], [0, -1]]))
    roberts_y = cv2.filter2D(gray_image, -1, np.array([[0, 1], [-1, 0]]))
    roberts_edges = np.sqrt(np.square(roberts_x) + np.square(roberts_y))
    
    # Prewitt edge detection
    prewitt_x = cv2.filter2D(gray_image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(gray_image, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt_edges = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    
    # Line edge detection
    kernel_line_x = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    kernel_line_y = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    line_edges_x = cv2.filter2D(gray_image, -1, kernel_line_x)
    line_edges_y = cv2.filter2D(gray_image, -1, kernel_line_y)
    line_edges = np.sqrt(np.square(line_edges_x) + np.square(line_edges_y))
    
    return roberts_edges, prewitt_edges, sobel_edges, line_edges

# Load an image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path)

# Apply edge detection
roberts_edges, prewitt_edges, sobel_edges, line_edges = edge_detection(image)

# Display original and edge-detected images
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(roberts_edges, cmap='gray')
plt.title('Roberts Edge Detection')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(prewitt_edges, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

plt.show()
