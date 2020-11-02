from preprocess_helpers import *
from globals import *


class ImageProcessor:
    """Singleton-like. All methods are static. Do not instantiate."""
    camera = None
    frame_count = 0
    #default_mask = None
    #mask = None
    #width = None
    #height = None
    #mask_vertices = None

    @classmethod
    def init(cls, size, camera, show_dbg=False):
        cls.camera = camera
        cls.frame_count = 0
        cls.txt1 = ""
        # cls.width, cls.height = size
        #cls.default_mask = generate_mask(cls.width, cls.height, 0.05, int(cls.height * 0.6))
        #cls.mask = cls.default_mask
        #cls.show_dbg = show_dbg


    @classmethod
    def reset(cls):
        cls.frame_count = 0
        #cls.mask = cls.default_mask
        pass

    @classmethod
    def do(cls, img, show_dbg=False):
        undistorted = cls.camera.undistort(img)
        binary = combined_threshold_3(undistorted, show_dbg and False)
        #masked = apply_mask(binary, cls.mask)
        birdseye = warp(binary, cls.camera.perspective_matrix, show_dbg and False)
        leftx, lefty, rightx, righty, windows_img = find_lane_pixels(birdseye, show_dbg and False)
        left_fit_px, right_fit_px, left_fit_m, right_fit_m = fit_polynomial(leftx, lefty, rightx, righty, windows_img, show_dbg and False)
        #poly_img = draw_polys_inplace(left_fit, right_fit, colored_lane_pixels, show_dbg and True)

        owerlay_birdseye = np.zeros_like(img)
        draw_polys_inplace(left_fit_px, right_fit_px, owerlay_birdseye, show_dbg and True)
        birdseye_extended = np.dstack((birdseye, birdseye, birdseye)) * 128
        white_patches = cv2.addWeighted(birdseye_extended, 1.0, owerlay_birdseye, 1.0, 0.)

        colors_on_green = cv2.addWeighted(white_patches, 1.0, windows_img, 1.0, 0.)
        owerlay_persp = warp(colors_on_green, cls.camera.perspective_inv_matrix, show_dbg and False)
        combined = cv2.addWeighted(img, 0.5, owerlay_persp, 1.0, 0.)

        # left_radius_px, right_radius_px = measure_radius_px(left_fit_px, right_fit_px)
        # cv2.putText(combined, "Left radius: {} px, right radius: {} px".format(left_radius_px, right_radius_px),
        #            (0, 40), cv2.QT_FONT_NORMAL, 1, color=(255, 255, 255))

        left_radius_m, right_radius_m = measure_radius_px(left_fit_m, right_fit_m)
        if cls.txt1 == None or cls.frame_count % 10 == 0:
            cls.txt1 = "Left radius: {:5.2f}km, right radius: {:5.2f}km".format(left_radius_m/1000., right_radius_m/1000.)
        cv2.putText(combined, cls.txt1, (0, 25), cv2.QT_FONT_NORMAL, 1, color=(255, 255, 255))

        cls.frame_count += 1
        return combined

    @classmethod
    def show(cls, img, title=None):
        colormap = None if len(img.shape) > 2 and img.shape[2] > 1 else 'gray'
        if title:
            plt.title(title)
        plt.imshow(img, cmap=colormap)
        plt.show()
