from preprocess_helpers import *
from line import Line
from line_history import LineHistory
from globals import *


class ImageProcessor:
    """Singleton-like. All methods are static. Do not instantiate."""
    camera = None
    frame_count = 0
    txt = ""
    left_history = None
    right_history = None
    #default_mask = None
    #mask = None
    #width = None
    #height = None
    #mask_vertices = None

    @classmethod
    def init(cls, size, camera, line_histories = None, show_dbg=False):
        cls.camera = camera
        cls.frame_count = 0
        cls.txt1 = ""
        if line_histories:
            cls.left_history = line_histories[0]
            cls.right_history = line_histories[1]
        # cls.width, cls.height = size
        #cls.default_mask = generate_mask(cls.width, cls.height, 0.05, int(cls.height * 0.6))
        #cls.mask = cls.default_mask
        #cls.show_dbg = show_dbg

    @classmethod
    def reset(cls):
        cls.frame_count = 0
        #cls.mask = cls.default_mask

    @classmethod
    def has_history(cls):
        return cls.left_history and cls.right_history

    @classmethod
    def sanity_checks(cls, left_line, right_line, left_fit_px, right_fit_px):
        ok_parallel, deviations = left_line.is_sane_parallel(right_line)
        ok_distance, min, max = left_line.is_sane_other_dist(right_line)
        strs = [
            'Parallel {} {}'.format(
                'OK' if ok_parallel else 'INSANE',
                'deviation between min and max: {:6d}px'.format(int(deviations))
            ),
            'Distance {} {}'.format(
                'OK' if ok_distance else 'INSANE',
                'min:{:6d}px max:{:6d}px'.format(int(min), int(max))
            ),
        ]
        ok_laverage, ok_raverage = 1, 1
        if cls.has_history():
            ok_laverage, lavg, ldiffs = cls.left_history.is_sane_avg(left_fit_px)
            ok_raverage, ravg, rdiffs = cls.right_history.is_sane_avg(right_fit_px)
            strs += [
                'Left history {} {}'.format(
                    'OK' if ok_laverage else 'INSANE',
                    'coeff diffs: {:8.7f}, {:5.3f}, {:5.1f}'.format(ldiffs[0], ldiffs[1], ldiffs[2])
                ),
                'Right history {} {}'.format(
                    'OK' if ok_raverage else 'INSANE',
                    'coeff diffs: {:8.7f}, {:5.3f}, {:5.1f}'.format(rdiffs[0], rdiffs[1], rdiffs[2])
                ),
            ]
        results = np.array([ok_parallel, ok_distance, ok_laverage, ok_raverage])
        ok_count = results.sum()
        merged = np.vstack((results, strs)).T
        return ok_count, len(results), merged

    @classmethod
    def sanity_to_img(cls, img, details):
        x, y, dy = 0, 75, 25
        colors = [(255, 128, 128), (128, 255, 128)]
        for is_ok, txt in details:
            cv2.putText(img, txt, (x, y), cv2.QT_FONT_NORMAL, 1, colors[int(is_ok)])
            y += dy

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
        cv2.putText(combined, 'Frame: {:d}'.format(cls.frame_count), (0, 50), cv2.QT_FONT_NORMAL, 1, color=(255, 255, 255))

        # sanity checks
        lline = Line(cls.frame_count, 'left', img.shape[0], left_fit_px, 1)
        rline = Line(cls.frame_count, 'right', img.shape[0], right_fit_px, 1)
        check_ok_count, check_total_count, sanity_details = cls.sanity_checks(lline, rline, left_fit_px, right_fit_px)
        cls.sanity_to_img(combined, sanity_details)

        if check_ok_count == check_total_count:
            lline.quality = 5
            rline.quality = 5
        else:
            lline.quality = 5 - (check_total_count - check_ok_count)
            rline.quality = 5 - (check_total_count - check_ok_count)
            # TODO: replace the fits and the images with predicted ones if exist

        cls.left_history.update(lline)
        cls.right_history.update(rline)

        cls.frame_count += 1
        return combined

    @classmethod
    def show(cls, img, title=None):
        colormap = None if len(img.shape) > 2 and img.shape[2] > 1 else 'gray'
        if title:
            plt.title(title)
        plt.imshow(img, cmap=colormap)
        plt.show()
