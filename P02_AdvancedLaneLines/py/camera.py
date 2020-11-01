import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


class Camera:
    def __init__(self, images_path, pattern_size, show_dbg=False):
        self.calibrate(images_path, pattern_size, show_dbg)

    def calibrate(self, images_path, pattern_size, show_dbg=False):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        images_paths = glob.glob(images_path)

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



        return {'matrix':[], 'distorsion':[], 'rotations':[], 'transforms':[]}


    #def undistort(self, image, )