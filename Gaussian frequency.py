import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_low_pass_filter(image, cutoff_frequency):
    # Compute the Fourier transform of the image
    f_transform = np.fft.fft2(image)
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Create a meshgrid of frequencies
    u, v = np.meshgrid(np.arange(-cols/2, cols/2), np.arange(-rows/2, rows/2))
    
    # Compute the distance from the center
    distance = np.sqrt(u**2 + v**2)
    
    # Create the Gaussian low pass filter
    gaussian_low_pass_filter = np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))
    
    # Apply the filter to the shifted Fourier transform
    f_transform_shifted_low_pass = f_transform_shifted * gaussian_low_pass_filter
    
    # Shift the zero frequency component back to the corner
    f_transform_low_pass = np.fft.ifftshift(f_transform_shifted_low_pass)
    
    # Compute the inverse Fourier transform
    image_low_pass = np.fft.ifft2(f_transform_low_pass).real
    
    return image_low_pass

# Load an image
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Perform Gaussian low pass filtering with cutoff frequency of 50
cutoff_frequency_low_pass = 50
image_low_pass = gaussian_low_pass_filter(image, cutoff_frequency_low_pass)

# Display original and filtered images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_low_pass, cmap='gray')
plt.title('Gaussian Low Pass Filtered Image')
plt.axis('off')

plt.show()
