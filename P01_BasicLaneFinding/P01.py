#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def multichannel_canny(img):
    channels = cv2.split(img)
    r = gaussian_blur(channels[0], 5)
    canny_r = cv2.Canny(r, 70, 150)
    g = gaussian_blur(channels[1], 5)
    canny_g = cv2.Canny(g, 70, 150)
    b = gaussian_blur(channels[2], 5)
    canny_b = cv2.Canny(b, 70, 150)

    # merge the three by taking the highest value for every pixel
    max_rg = cv2.max(canny_r, canny_g)
    max_rgb = cv2.max(max_rg, canny_b)
    return max_rgb


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines_inplace(img, lines, color=[255,0,0], thickness=2):
    for line in lines:
        cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it. Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing. The result image is computed as follows: initial_img * alpha + img * beta + gamma
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

import os
os.listdir("test_images/")


def generate_mask_vertices(img_width, img_height, top_half_width, horizon):
    x_top_left = img_width * (0.5 - top_half_width)
    x_top_right = img_width * (0.5 + top_half_width)
    vertices = np.array([[(0, img_height), (x_top_left, horizon), (x_top_right, horizon), (img_width, img_height)]], dtype=np.int32)
    return vertices


def extend_line(line, img_width, img_height, horizon):
    """Generates a line that crosses the given point with the given slope,
    and that reaches both the bottom of the image and the virtual horizon.

    Arguments:
        line: type: dict
            center: dict of two floats or None
            slope: float or None
            line == None if missing

    Returns: a line that preserves its direction and that is extended if needed to y_top.
    """
    if line == None or line["slope"] == 0:
        return [[0, 30, img_width, 30]] # a big line for debugging purposes
        # return None

    x0, y0 = line["center"]
    m = line["slope"]

    # calculate the bottom point
    dx = (y0 - img_height) / m
    x1 = x0 + dx
    y1 = img_height

    # calculate the top point
    dx = (y0 - horizon) / m
    x2 = x0 + dx
    y2 = horizon

    return [[int(x1), int(y1), int(x2), int(y2)]]


def get_main_lines(lines):
    """Finds the left and right main lines.

    Arguments:
        lines: list of lines, where:
        each line is represented like this: [[x1, y1, x2, y2]]
        NOTE that line shape is (1, 4) and not (4)!

    Returns: a tuple of two main lines or Nones if not found.
        Each main line contains:
        'center': [x, y] floats
        'slope' : float
    """
    min_degree, max_degree = 20., 70. # absolute angles in degrees from the horizontal
    main_line_left, main_line_right = None, None

    # right and left m values (slopes) and center coordinates
    rm = []
    lm = []
    rc = []
    lc = []

    # assign lines to the left or to the right side
    # filter out lines which don't meet the angle limitations
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if dx != 0:
            slope = (y1 - y2) / dx # y1 and y2 is switched because y grows from top to bottom
            deg = 180 * math.atan(slope) / np.pi
            if abs(deg) >= min_degree and abs(deg) <= max_degree:
                # add to left or right lists
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if deg > 0:
                    lm.append(slope)
                    lc.append([center_x, center_y])
                else:
                    rm.append(slope)
                    rc.append([center_x, center_y])

    # find the average slopes and centers
    if len(lm) > 0:
        main_line_left = {}
        main_line_left["slope"] = sum(lm) / len(lm)
        main_line_left["center"] = np.average(np.array(lc), axis=0)

    if len(rm) > 0:
        main_line_right = {}
        main_line_right["slope"] = sum(rm) / len(rm)
        main_line_right["center"] = np.average(np.array(rc), axis=0)

    return [main_line_left, main_line_right]


class History:
    def __init__(self, size):
        self.size = size # max number of historical records
        self.reset()

    def reset(self):
        self.queue = [None] * self.size
        self.top = 0 # the position after the most recent data, position to write to
        self.bottom = 0 # the position of the oldest data

    def update(self, lines):
        if self.queue[self.top] == None:
            self.queue[self.top] = [{"center" : [0., 0.], "slope" : 0.}, {"center" : [0., 0.], "slope" : 0.}]
        self.queue[self.top][0]["center"][0] = lines[0]["center"][0] # left x
        self.queue[self.top][0]["center"][1] = lines[0]["center"][1] # left y
        self.queue[self.top][0]["slope"]     = lines[0]["slope"]     # left slope
        self.queue[self.top][1]["center"][0] = lines[1]["center"][0] # right x
        self.queue[self.top][1]["center"][1] = lines[1]["center"][1] # right y
        self.queue[self.top][1]["slope"]     = lines[1]["slope"]     # right slope
        self.top = self._get_next_idx(self.top)
        self._update_bottom(self.top)
        retval = self._get_avg()
        return retval

    def _get_next_idx(self, idx):
        idx = 0 if idx >= self.size - 1 else idx + 1 # grow until possible, then start from 0
        return idx

    def _update_bottom(self, idx):
        if self.top == self.bottom: # top reached bottom, so bottom needs to move
            self.bottom = self._get_next_idx(self.bottom)

    def _get_avg(self):
        # sums of left and right line coordinates and slopes
        sum_lx, sum_ly, sum_lm, sum_rx, sum_ry, sum_rm = 0, 0, 0, 0, 0, 0
        idx = self.bottom
        count = 0
        while (True):
            if self.queue[idx] == None: # no data yet, should never happen
                break
            else:
                count += 1
                left = self.queue[idx][0]
                right = self.queue[idx][1]
                sum_lx += left["center"][0]
                sum_ly += left["center"][1]
                sum_lm += left["slope"]
                sum_rx += right["center"][0]
                sum_ry += right["center"][1]
                sum_rm += right["slope"]

                if idx == self.top: # reached the end
                    break

                idx = self._get_next_idx(idx)

        return [{"center" : [sum_lx / count, sum_ly / count], "slope" : sum_lm / count},
                {"center" : [sum_rx / count, sum_ry / count], "slope" : sum_rm / count}]


def process_image(image):
    height, width = image.shape[0], image.shape[1]
    y_horizon = int(height * 0.6)
    half_top_percent = 0.03 # half width of the top of the trapezoid in screen width percent

    img_cannied = multichannel_canny(image)

    # generate mask
    mask_vertices = generate_mask_vertices(width, height, half_top_percent, y_horizon)
    img_masked = region_of_interest(img_cannied, mask_vertices)

    # draw horizon
    # tmp_img = np.zeros((height, width, 3), dtype=np.uint8)
    # cv2.line(tmp_img, (0, y_horizon), (width, y_horizon), [128, 128, 255] , 5)
    # img_combined = weighted_img(tmp_img, image, alpha=1.0, beta=0.5, gamma=0.)

    # get hough lines
    rho = 1                 # distance resolution in pixels of the Hough grid (1)
    theta = 1 * np.pi/180   # angular resolution in radians of the Hough grid (np.pi/180)
    threshold = 35          # minimum number of votes (intersections in Hough grid cell) (35)
    min_line_length = 20    #minimum number of pixels making up a line (5)
    max_line_gap = 2        # maximum gap in pixels between connectable line segments (2)
    lines = hough_lines(img_masked, rho, theta, threshold, min_line_length, max_line_gap)
    tmp_img = np.zeros((height, width, 3), dtype=np.uint8)
    # draw_lines_inplace(tmp_img, lines, color=[0, 0, 255], thickness=5)
    img_combined = weighted_img(tmp_img, image, alpha=1.0, beta=0.5, gamma=0.)

    # get two main lines out of the multitude of small lines
    main_lines = get_main_lines(lines)
    historical_main_lines = history.update(main_lines) # update the history and get averaged lines back

    # extend lines from virtual horizon to the bottom of the image
    tmp_img = np.zeros((height, width, 3), dtype=np.uint8)
    for line in historical_main_lines:
        extended_line = extend_line(line, width, height, y_horizon)
        cv2.line(tmp_img, (extended_line[0][0], extended_line[0][1]), (extended_line[0][2], extended_line[0][3]), [255, 0, 0], 8)
    img_combined = weighted_img(tmp_img, img_combined, alpha=1.0, beta=0.5, gamma=0.)

    return img_combined


def test_on_still_images():
    for image_name in os.listdir("test_images/"):
        path_name = "test_images/" + image_name
        print('processing image:', path_name )
        source = mpimg.imread(path_name)
        processed = process_image(source)
        mpimg.imsave("test_images_output/" + image_name, processed)
        plt.imshow(processed)
        plt.show()


# ## uncomment to generate the processed still images
# test_on_still_images()


from moviepy.editor import VideoFileClip

history = History(size=8)
white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output)
white_clip.close()
history.reset()

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
#clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output)
yellow_clip.close()
history.reset()

challenge_output = 'test_videos_output/challenge.mp4'
#clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output)
challenge_clip.close()
