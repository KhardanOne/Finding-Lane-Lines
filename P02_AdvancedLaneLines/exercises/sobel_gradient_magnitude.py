import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Read in an image
image = mpimg.imread('../input/exercise_images/signs_vehicles_xygrad.png')


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=5, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx_fl = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely_fl = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, sobel_kernel)

    # 3) Calculate the magnitude
    sobelxy_fl = np.sqrt( (sobelx_fl**2) + (sobely_fl**2) )

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    img_scaled = np.uint8(255 * sobelxy_fl / np.max(sobelxy_fl))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(img_gray)
    binary_output[(img_scaled >= mag_thresh[0]) & (img_scaled <= mag_thresh[1])] = 1

    return binary_output


# Run the function
mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(60, 255))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()