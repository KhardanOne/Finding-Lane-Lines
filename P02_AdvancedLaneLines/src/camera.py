import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os


class Camera:

    def __init__(self, images_path, pattern_size, show_dbg=False):
        matrix = None
        distortion_coeffs = None
        self.calibrate(images_path, pattern_size, show_dbg)

    def calibrate(self, images_path, pattern_size, show_dbg=False):
        print('calibrationg camera...', end=" ")
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        images_paths = glob.glob(images_path)

        # determine image size
        imgshape = cv2.imread(images_paths[0]).shape

        # Step through the list and search for chessboard corners
        for fname in images_paths:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                if show_dbg:
                    cv2.imshow('img',img)
                    cv2.waitKey(350)

        if show_dbg:
            cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (imgshape[1], imgshape[0]), None, None)
        if ret:
            self.matrix = mtx
            self.distortion_coeffs = dist
        else:
            assert False, 'Calibration failed'

        print('done.')
        return ret, mtx, dist, rvecs, tvecs


    def generate_test_images(self, images_path, output_dir):
        """images_paths can contain * wildcard."""
        print('generating test images...', end=" ")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        images_paths = glob.glob(images_path)
        for fname in images_paths:
            src = cv2.imread(fname)
            dst = self.undistort(src)
            path = output_dir + os.path.basename(fname)
            plt.imsave(path, dst)
        print('done.')


    def undistort(self, image):
        try:
            undistorted = cv2.undistort(image, self.matrix, self.distortion_coeffs)
            return undistorted
        except Exception as e:
            print('undistortion error:', str(e))
