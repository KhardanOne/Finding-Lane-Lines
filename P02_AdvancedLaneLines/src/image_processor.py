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

    @classmethod
    def init(cls, size, camera, show_dbg=False):
        cls.width, cls.height = size
        cls.camera = camera
        cls.default_mask = generate_mask(cls.width, cls.height, 0.05, int(cls.height * 0.6))
        cls.mask = cls.default_mask
        cls.show_dbg = show_dbg

    @classmethod
    def reset(cls):
        cls.mask = cls.default_mask

    @classmethod
    def do(cls, img):
        undistorted = cls.camera.undistort(img)
        #binary = combined_threshold_3(undistorted, show_dbg=cls.show_dbg)
        binary = combined_threshold_3(undistorted, False)
        #masked = apply_mask(binary, cls.mask)
        birdseye = remove_perspective(binary)
        if cls.show_dbg:
            cls.show(birdseye)

    @classmethod
    def show(cls, img):
        colormap = None if len(img.shape) > 2 and img.shape[2] > 1 else 'gray'
        plt.imshow(img, cmap=colormap)
        plt.show()
