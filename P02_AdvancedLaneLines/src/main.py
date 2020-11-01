from camera import Camera

# config
calibration_image_path = '../input/camera_cal/calibration*.jpg'
calibration_test_target_dir = '../output/image_test_undistortion/'


def main():
    camera = Camera(calibration_image_path, pattern_size=(9, 6), show_dbg=False)
    camera.generate_test_images(calibration_image_path, calibration_test_target_dir)



if __name__ == '__main__':
    main()
