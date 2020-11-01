# TODO: finish or delete

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('../input/exercise_images/signs_vehicles_xygrad.png')


def abs_sobel_thresh(gray_img, orient='x', kernel_size=5, thresh=(0, 255)):
    """Applies Sobel in x or y direction."""
    if orient == 'x':
        deriv = (1, 0)
    elif orient == 'y':
        deriv = (0, 1)
    else:
        assert False, 'invalid orient: {}'.format(orient)

    img_sobel_float = cv2.Sobel(gray_img, cv2.CV_64F, deriv[0], deriv[1], ksize=kernel_size)
    img_sobel_float = np.absolute(img_sobel_float)
    img_scaled_sobel = np.uint8(255 * img_sobel_float / np.max(img_sobel_float))
    binary_output = np.zeros_like(img_scaled_sobel)
    binary_output[(img_scaled_sobel >= thresh[0]) & (img_scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(gray_img, kernel_size=5, mag_thresh=(0, 255)):
    """Applies Sobel x and y, then computes their magnitude."""
    sobelx_fl = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, kernel_size)
    sobely_fl = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, kernel_size)
    sobelxy_fl = np.sqrt( (sobelx_fl**2) + (sobely_fl**2) )
    img_scaled = np.uint8(255 * sobelxy_fl / np.max(sobelxy_fl))
    binary_output = np.zeros_like(gray_img)
    binary_output[(img_scaled >= mag_thresh[0]) & (img_scaled <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(gray_img, kernel_size=3, thresh=(0, np.pi / 2)):
    """Applies Sobel x and y, then computes the dir of gradient. 0 means vertical."""
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(dir)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return binary_output


def bin_logical_or(*gray_bins):
    """Combines binary results. Inputs most be of same size."""
    combined = np.zeros_like(gray_bins[0])
    for bin in gray_bins:
        combined = np.logical_or(combined, bin)
    return combined


def combined_threshold_1(color_img, show_dbg=False):
    """Applies different methods to find the thresholds in the image."""
    gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    r, g, b = cv2.split(color_img)
    hls = cv2.cvtColor(color_img, cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)

    bin_gradx = abs_sobel_thresh(gray, 'x', 5, (25, 255))
    bin_grady = abs_sobel_thresh(gray, 'y', 5, (25, 255))
    bin_dir = dir_threshold(gray, kernel_size=15, thresh=(40 * np.pi / 180, 75 * np.pi / 180))
    bin_combined = bin_logical_or(bin_gradx, bin_grady, bin_dir)

    if show_dbg:
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 6))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.imshow(color_img)
        ax2.set_title('Sobel X')
        ax2.imshow(bin_gradx, cmap='gray')
        ax3.set_title('Sobel Y')
        ax3.imshow(bin_grady, cmap='gray')
        ax4.set_title('Thresholded Grad. Dir.')
        ax4.imshow(bin_dir, cmap='gray')
        ax5.set_title('Combined')
        ax5.imshow(bin_combined, cmap='gray')
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()


combined_threshold_1(image, True)
