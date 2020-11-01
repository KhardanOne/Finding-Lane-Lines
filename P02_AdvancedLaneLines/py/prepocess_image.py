# TODO: finish or delete

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#image = mpimg.imread('../input/exercise_images/signs_vehicles_xygrad.png')
image = mpimg.imread('../input/exercise_images/bridge_shadow.jpg')


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

def bin_count(threshold, *gray_bins):
    """Pixelwise count the hits in all of the inputs. Result contains true if threshold is reached."""
    count = np.zeros_like(gray_bins[0])
    for bin in gray_bins:
        count = np.add(count, bin)
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
    s = hls[:,:,2]
    bin_s_gradx = abs_sobel_thresh(s, 'x', 3, (25, 255))

    thresh_deg = (40, 75)
    thresh_rad = (thresh_deg[0] * np.pi / 180, thresh_deg[1] * np.pi / 180)
    bin_r_dir = dir_threshold(r, kernel_size=15, thresh=thresh_rad)
    bin_g_dir = dir_threshold(g, kernel_size=15, thresh=thresh_rad)
    bin_b_dir = dir_threshold(b, kernel_size=15, thresh=thresh_rad)
    bin_s_dir = dir_threshold(s, kernel_size=15, thresh=thresh_rad)

    bin_combined = bin_count(4, bin_r_gradx, bin_g_gradx, bin_b_gradx, bin_s_gradx,
                                bin_r_dir, bin_g_dir, bin_b_dir, bin_s_dir)

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
        ax12.set_title('Combined')
        ax12.imshow(bin_combined, cmap='gray')
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return bin_combined

#combined_threshold_1(image, True)
combined_threshold_2(image, show_dbg=True)
