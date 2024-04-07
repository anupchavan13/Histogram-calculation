import cv2
import numpy as np
import pywt

# Load an image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the wavelet family and level of decomposition
wavelet = 'haar'  # You can change this to other wavelet families like 'db1', 'db2', 'db3', etc.
level = 3  # You can change the level of decomposition

# Perform Discrete Wavelet Transform (DWT)
coeffs = pywt.wavedec2(image, wavelet, level=level)

# Extract features from DWT coefficients
features = []
for coeff in coeffs:
    features.extend(coeff.ravel())

# Convert the feature list to a numpy array
features = np.array(features)

# Print the shape of the features array
print("Shape of extracted features:", features.shape)
