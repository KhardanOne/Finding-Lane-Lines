from camera import Camera

# config
calibration_image_path = '../input/camera_cal/calibration*.jpg'



def main():
    camera = Camera(calibration_image_path, pattern_size=(9, 6), show_dbg=True)



if __name__ == '__main__':
    main()
