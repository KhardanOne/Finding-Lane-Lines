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
    def do_orig_persp(cls, img, verbose=False, show_dbg=False):
        undistorted = cls.camera.undistort(img)
        binary = combined_threshold_3(undistorted, show_dbg and False)
        # masked = apply_mask(binary, cls.mask)
        return undistorted, binary

    @classmethod
    def extend(cls, img, x_pad_mult, verbose=False, show_dbg=False):
        """Example: 2) means the height will be the same, but the width will be 2 + 1 + 2 times wider than original"""
        x_scale = 1 + 2 * x_pad_mult
        if len(img.shape) > 2:  #  not grayscale, nor binary
            shape = (img.shape[0], int(img.shape[1] * (1 + x_pad_mult)), img.shape[2])
        else:
            shape = (img.shape[0], int(img.shape[1] * (1 + x_pad_mult)))
        x_transform = int(img.shape[1] * x_pad_mult)
        img_extd = np.zeros(shape)
        img_extd[0, x_transform] = img
        if verbose:
            thickness_side, thickness_top, color = 20, 100, (255, 0, 0)
            top, left, bottom, right = 0, 0, img.shape[0] - 1, img.shape[1] -1
            cv2.line(img, (left, top), (left, bottom), color, thickness_side)
            cv2.line(img, (right, top), (right, bottom), color, thickness_side)
            cv2.line(img, (left, top), (right, bottom), color, thickness_top)
        if show_dbg:
            plt.imshow(img)
            plt.show()
        return img_extd, x_transform, x_scale

    @classmethod
    def do_2d(cls, bin_img, verbose=False, show_dbg=False):
        leftx, lefty, rightx, righty, bin_out = find_lane_pixels(bin_img, verbose=verbose and True,
                                                                 show_dbg=show_dbg and True)
        left_fit_px, right_fit_px, left_fit_m, right_fit_m = fit_polynomial(leftx, lefty, rightx, righty, bin_out,   ### TODO: coords are also modified by padding, follow it
                                                                            verbose=verbose and True,
                                                                            show_dbg=show_dbg and False)
        #poly_img = draw_polys_inplace(left_fit, right_fit, colored_lane_pixels, show_dbg and True)
        raw_xy = (leftx, lefty, rightx, righty)
        fit_px, fit_m = (left_fit_px, right_fit_px), (left_fit_m, right_fit_m)
        return raw_xy, fit_px, fit_m, bin_windows
        pass

    @classmethod
    def do_2d_overlay(cls, bin_img, raw_xy, fit_px, bin_windows, size2d, verbose=False, show_dbg=False):
        bin = np.zeros_like(bin_img)
        img = np.dstack((bin, bin, bin)) * 128
        draw_polys_inplace(fit_px[0], fit_px[1], bin, show_dbg and True)

        white_patches = cv2.addWeighted(birdseye_extended, 1.0, owerlay_birdseye, 1.0, 0.)
        colors_on_green = cv2.addWeighted(white_patches, 1.0, windows_img, 1.0, 0.)
        pass

    @classmethod
    def do_persp(cls, img, verbose=False, show_dbg=False):
        pass

    @classmethod
    def do_persp_overlay(cls, img, verbose=False, show_dbg=False):
        pass

    @classmethod
    def do(cls, img, verbose=False, show_dbg=False):  # verbose: extra info on the video; show_dbg: extra info in window
        # processing in orig perspective
        img_orig_persp_corr, bin_orig_persp_corr = cls.do_orig_persp(img, verbose, show_dbg and True)

        # converting to 2D
        want_to_extend = True  ######################################################################################### tune here
        x_trfm, x_scale = 0, 1.  # for pixel coord calculation transform and scale
        if want_to_extend:
            x_pad_mult = 1  # x will be padded x_pad_mult * widht on the left and with the same amount on the right #### tune here
            bin_img, x_trfm, x_scale = cls.extend(bin_orig_persp_corr, x_pad_mult)                                      ### TODO: check if identical
        warped_to_2D = warp(bin_orig_persp_corr, cls.camera.perspective_matrix, show_dbg = show_dbg and True)

        # work in 2D
        bin2d = cls.do_2d(warped_to_2D, verbose=verbose and True, show_dbg=show_dbg and True)
        img2d_overlay, img2d_for_persp = cls.do_2d_overlay(bin2d, verbose=verbose and True, show_dbg=show_dbg and True)

        # processing again in perspective


        # convert these:

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
