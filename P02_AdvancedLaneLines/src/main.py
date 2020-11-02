from globals import *
import matplotlib.image as mpimg
from camera import Camera
# from preprocess_helpers import *
from image_processor import *
import numpy as np

# configure settings here
CFG['camera_calib_file'] = '../camera_calib.json'
CFG['calibration_image_path'] = '../input/camera_cal/calibration*.jpg'
CFG['calibration_test_target_dir'] = '../output/image_test_undistortion/'
CFG['perspective_calib_img'] = '../input/road_images/straight_lines2.jpg'
# starting from bottom left cw, xy
CFG['perspective_calib_pts'] = np.float32([[271, 682], [593, 450], [689, 450], [1048, 682]])
#CFG['perspective_calib_pts'] = np.float32([[271, 682], [614, 435], [666, 435], [1048, 682]])

# because it is not possible to pass data to callbacks, ugly globals are needed
GLOBAL['camera'] = None
GLOBAL['frame_cnt'] = 0


def main():
    GLOBAL['camera'] = Camera(CFG['calibration_image_path'],
                              pattern_size=(9, 6),
                              save_path=CFG['camera_calib_file'],
                              persp_path=CFG['perspective_calib_img'],
                              persp_ref_points=CFG['perspective_calib_pts'],
                              show_dbg=True,
                              )
    # camera.generate_test_images(CFG['calibration_image_path'], CFG['calibration_test_target_dir'])

    img_source = mpimg.imread('../input/road_images/test4.jpg')
    ImageProcessor.init((img_source.shape[1], img_source.shape[0]), GLOBAL['camera'], show_dbg=True)
    ImageProcessor.do(img_source, show_dbg=True)


if __name__ == '__main__':
    main()


# TODO: forget the old masking method... use unwarped rect instead
