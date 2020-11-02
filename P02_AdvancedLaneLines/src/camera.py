import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os


class Camera:

    def __init__(self, images_path, pattern_size, show_dbg=False, save_path=None):
        self.matrix = None
        self.distortion_coeffs = None

        # Try to load the matrix and distortion_coeffs from file.
        if save_path:
            success = self.load(save_path)
            if success:
                return

        # If failed to load: generate and save
        self.calibrate(images_path, pattern_size, show_dbg)
        self.save(save_path)

    def save(self, path):
        """Saves the camera matrix and distortion coefficients to the file."""
        print('Saving to file:', path, end=" ")
        file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        file.write("mtx", self.matrix)
        file.write("dist", self.distortion_coeffs)
        file.release()
        print('done.')

    def load(self, path):
        """Loads the camera matrix and distortion coefficients from the file."""
        print('Loading from file:', path, end=" ")
        if os.path.exists(path):
            file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            self.matrix = file.getNode("mtx").mat()
            self.distortion_coeffs = file.getNode("dist").mat()
            file.release()
            print('done.')
            return True
        else:
            print("failed. File doesn't exist.")
            return False

    def calibrate(self, images_path, pattern_size, show_dbg=False):
        print('Calibrating...', end=" ")
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        images_paths = glob.glob(images_path)

        # determine image size
        imgshape = cv2.imread(images_paths[0]).shape

        # Step through the list and search for chessboard corners
        for fname in images_paths:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                if show_dbg:
                    cv2.imshow('img', img)
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

    def undistort(self, image):
        try:
            undistorted = cv2.undistort(image, self.matrix, self.distortion_coeffs)
            return undistorted
        except Exception as e:
            print('Undistortion error:', str(e))

    def generate_test_images(self, images_path, output_dir):
        """images_paths can contain * wildcard."""
        print('Generating camera undistortion test images...', end=" ")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        images_paths = glob.glob(images_path)
        for fname in images_paths:
            src = cv2.imread(fname)
            dst = self.undistort(src)
            path = output_dir + os.path.basename(fname)
            plt.imsave(path, dst)
        print('done.')
