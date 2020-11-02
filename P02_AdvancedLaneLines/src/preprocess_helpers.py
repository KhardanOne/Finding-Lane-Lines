import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from globals import *

# image = mpimg.imread('../input/exercise_images/signs_vehicles_xygrad.png')
image = mpimg.imread('../input/exercise_images/bridge_shadow.jpg')


def multichannel_canny(img):
    """Separates the image to three channels. Applies Gaussian Blur and Canny for each channels separately, then
    combines them taking the maximum value for every pixel. Uses fixed and predefined parameters for those operations.

    Arguments:
        img: An image with three layers, e.g. RGB, BGR, HLS, HSV, etc.

    Returns:
        A grayscale image of the same size.
    """
    channels = cv2.split(img)
    r = cv2.GaussianBlur(channels[0], (5, 5), 0)
    g = cv2.GaussianBlur(channels[1], (5, 5), 0)
    b = cv2.GaussianBlur(channels[2], (5, 5), 0)
    canny_r = cv2.Canny(r, 70, 150)
    canny_g = cv2.Canny(g, 70, 150)
    canny_b = cv2.Canny(b, 70, 150)

    max_rg = cv2.max(canny_r, canny_g)
    max_rgb = cv2.max(max_rg, canny_b)
    return max_rgb


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
    sobelxy_fl = np.sqrt((sobelx_fl ** 2) + (sobely_fl ** 2))
    img_scaled = np.uint8(255 * sobelxy_fl / np.max(sobelxy_fl))
    binary_output = np.zeros_like(gray_img)
    binary_output[(img_scaled >= mag_thresh[0]) & (img_scaled <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(gray_img, kernel_size=3, thresh=(0, np.pi / 2)):
    """Applies Sobel x and y, then computes the dir of gradient. 0 means vertical."""
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output


def bin_logical_or(*gray_bins):
    """Combines binary results. Inputs most be of same size."""
    combined = np.zeros_like(gray_bins[0])
    for binary in gray_bins:
        combined = np.logical_or(combined, binary)
    return combined


def bin_count(threshold, *gray_bins):
    """Pixelwise count the hits in all of the inputs. Result contains true if threshold is reached."""
    count = np.zeros_like(gray_bins[0])
    for binary in gray_bins:
        count = np.add(count, binary)
    hits = np.zeros_like(gray_bins[0])
    hits[count >= threshold] = 1
    return hits


def combined_threshold_1(color_img, show_dbg=False):
    """Applies different methods to find the thresholds in the image."""
    gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
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

    return bin_combined


def combined_threshold_2(color_img, show_dbg=False):
    """Applies different methods to find the thresholds in the image."""
    color_img = cv2.GaussianBlur(color_img, (5, 5), 0)
    r, g, b = cv2.split(color_img)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)
    bin_r_gradx = abs_sobel_thresh(r, 'x', 3, (15, 255))
    bin_g_gradx = abs_sobel_thresh(g, 'x', 3, (15, 255))
    bin_b_gradx = abs_sobel_thresh(b, 'x', 3, (15, 255))

    hls = cv2.cvtColor(color_img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    bin_s_gradx = abs_sobel_thresh(s, 'x', 3, (25, 255))

    thresh_deg = (40, 75)
    thresh_rad = (thresh_deg[0] * np.pi / 180, thresh_deg[1] * np.pi / 180)
    bin_r_dir = dir_threshold(r, kernel_size=15, thresh=thresh_rad)
    bin_g_dir = dir_threshold(g, kernel_size=15, thresh=thresh_rad)
    bin_b_dir = dir_threshold(b, kernel_size=15, thresh=thresh_rad)
    bin_s_dir = dir_threshold(s, kernel_size=15, thresh=thresh_rad)

    bin_combined = bin_count(4, bin_r_gradx, bin_g_gradx, bin_b_gradx, bin_s_gradx,
                             bin_r_dir, bin_g_dir, bin_b_dir, bin_s_dir)

    mcc = multichannel_canny(color_img)

    if show_dbg:
        f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(16, 6))
        f.tight_layout()
        ax1.set_title('Original')
        ax1.imshow(image)
        ax2.set_title('R grad x')
        ax2.imshow(bin_r_gradx, cmap='gray')
        ax3.set_title('G grad x')
        ax3.imshow(bin_g_gradx, cmap='gray')
        ax4.set_title('B grad x')
        ax4.imshow(bin_b_gradx, cmap='gray')
        ax5.set_title('S grad x')
        ax5.imshow(bin_s_gradx, cmap='gray')
        ax6.set_title('Bin R direction')
        ax6.imshow(bin_r_dir, cmap='gray')
        ax7.set_title('Bin G direction')
        ax7.imshow(bin_g_dir, cmap='gray')
        ax8.set_title('Bin B direction')
        ax8.imshow(bin_b_dir, cmap='gray')
        ax9.set_title('Bin S direction')
        ax9.imshow(bin_s_dir, cmap='gray')
        ax10.set_title('Multichannel Canny')
        ax10.imshow(mcc, cmap='gray')
        ax12.set_title('Combined')
        ax12.imshow(bin_combined, cmap='gray')
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return bin_combined


def combined_threshold_3(color_img, show_dbg=False):
    """Applies different methods to find the thresholds in the image."""
    hls = cv2.cvtColor(color_img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min, thresh_max = 20, 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min, s_thresh_max = 170, 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Plotting thresholded images
    if show_dbg:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')

        mcc = multichannel_canny(color_img)

        ax3.set_title('Multichannel Canny')
        ax3.imshow(mcc, cmap='gray')

        plt.show()

    return combined_binary


def get_perspective_transform(img, show_dbg=False):
    pass


def remove_perspective(img, show_dbg=False):
    print(CFG['camera_calib_file'])
    return np.copy(img)


def region_of_interest(img, vertices):
    """Applies a mask defined by vertices to an image.

    Arguments:
        img: Source image of 1..4 channels
        vertices: Vertices that define a polygon.

    Returns:
        An image of the same size and channels. All pixels outside the polygon defined by vertices are black.
    """
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def generate_mask(img_width, img_height, top_half_width, horizon):
    """Generates a trapezoid mask. The trapezoid consists of ones, bacground of zeros.

    Arguments:
        img_width: Width of the image in pixels.
        img_height: Height of the image in pixels.
        top_half_width: Half-width of the trapezoid top edge. Given in screen width ratio.
            E.g. 0.05 means that the width of the trapezoid top will be 10% of the image width.
        horizon: The trapezoid top edge's y coordinate in pixels.

    Returns:
        A numpy array of 4 vertices, shape: (1, 4), dtype=np.int32.
    """
    x_top_left = img_width * (0.5 - top_half_width)
    x_top_right = img_width * (0.5 + top_half_width)
    vertices = np.array([[(0, img_height), (x_top_left, horizon), (x_top_right, horizon), (img_width, img_height)]],
                        dtype=np.int32)

    mask = np.zeros(img_width * img_height, dtype=np.uint8).reshape(img_height, -1)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, 255)

    return mask


def apply_mask(img, mask):
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask = np.dstack((mask, mask, mask)) if channel_count == 3 else np.dstack((mask, mask, mask, mask))

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
