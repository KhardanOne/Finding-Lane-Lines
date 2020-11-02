from numpy import np

class Line:

    def __init__(self, frame, detected, fit, allx, ally):
        self.frame = None                       # in which frame was the line added
        self.detected = False                   # if the polynomial coefficients are successfully identified
        self.fit = [np.array([False])]          # polynomial coefficients
        self.allx = None                        # x values for detected line pixels
        self.ally = None                        # y values for detected line pixels

        # # was the line detected in the last iteration?
        # self.detected = False
        # # x values of the last n fits of the line
        # self.recent_xfitted = []
        # #average x values of the fitted line over the last n iterations
        # self.bestx = None
        # #polynomial coefficients averaged over the last n iterations
        # self.best_fit = None
        # #polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]
        # #radius of curvature of the line in some units
        # self.radius_of_curvature = None
        # #distance in meters of vehicle center from the line
        # self.line_base_pos = None
        # #difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float')
        # #x values for detected line pixels
        # self.allx = None
        # #y values for detected line pixels
        # self.ally = None

