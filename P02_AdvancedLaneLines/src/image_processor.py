from preprocess_helpers import *
from globals import *


class ImageProcessor:
    """Singleton-like. All methods are static. Do not instantiate."""
    camera = None
    default_mask = None
    mask = None
    width = None
    height = None
    show_dbg = False
    mask_vertices = None
    persp_mtx = None

    @classmethod
    def init(cls, size, camera, show_dbg=False):
        cls.width, cls.height = size
        cls.camera = camera
        #cls.default_mask = generate_mask(cls.width, cls.height, 0.05, int(cls.height * 0.6))
        cls.mask = cls.default_mask
        cls.show_dbg = show_dbg


    @classmethod
    def reset(cls):
        cls.mask = cls.default_mask

    @classmethod
    def do(cls, img, show_dbg=False):
        undistorted = cls.camera.undistort(img)
        binary = combined_threshold_3(undistorted, show_dbg and False)
        #masked = apply_mask(binary, cls.mask)
        birdseye = warp(binary, cls.camera.perspective_matrix, show_dbg and False)
        leftx, lefty, rightx, righty, windows_img = find_lane_pixels(birdseye, show_dbg and False)
        left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty, windows_img, show_dbg and True)
        pass

    @classmethod
    def show(cls, img, title=None):
        colormap = None if len(img.shape) > 2 and img.shape[2] > 1 else 'gray'
        if title:
            plt.title(title)
        plt.imshow(img, cmap=colormap)
        plt.show()
