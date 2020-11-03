import numpy as np
from globals import *

class LineHistory:
    def __init__(self, side):
        self.side = side
        self.reset()

    def reset(self):
        self.history = [None] * (CFG['history_length'] + 1)  # +1 to allow for easier iterating over the next items

    def get_future_coeffs(self, for_frame):
        """
        !!! Not implemented !!!
        Calculates fit from old fit and old derived fit.
        Fails if the history doesn't contain enough data for the time periods.
        """
        raise NotImplemented
        pass

    def get_avg_coeffs(self, quality_limit=4):
        """Returns True if avg is found, False otherwise. Second return value is the coeffs np.array."""
        sums = np.array([0] * 3, dtype=np.float64)
        count = 0
        for line in self.history:
            if line is not None and line.quality >= quality_limit:
                sums += line.coeffs
                count += 1
        if count > 0:
            avg = sums / count
            return True, avg
        else:
            return False, np.array([0., 0., 80])

    def get_raw_xy(self):
        """!!! Not implemented !!! Return last x y raw data. Improved version: use derivatives for prediction."""
        raise NotImplemented
        pass

    def is_sane_avg(self, future_fit):
        """False if it deviates from the stored avg. Returns True if there is no enough data or it is OK."""
        if self._get_size() < 5:
            return True, np.array([0., 0., 0.]), np.array([0., 0., 0.])
        limits = np.array([0.0003, 0.5, 180.])  ############################################################# tune here
        has_avg, avg = self.get_avg_coeffs()
        if not has_avg:  # not enough data yet
            return True, np.array([0., 0., 0.]), np.array([0., 0., 0.])
        diffs = np.abs(future_fit - avg)
        result = 1 if np.less_equal(diffs, limits).all() else 0
        return result, avg, diffs

    def is_sane_future(self, future_fit):
        """!!! Not implemented !!! Like is_sane_avg but it also takes into account the deried fit."""
        raise NotImplemented
        pass

    def update(self, line):
        """Stores the line and returns the stored average."""
        line.side = line.side if line.side else self.side
        self._shift_stored_data()
        self.history[0] = line
        avg = self.get_avg_coeffs()
        return avg

    def _get_size(self):
        count = 0
        for i in range(CFG['history_length']):  # optimization possibility: store the size
            if self.history[i]:
                count += 1
        return count

    def _shift_stored_data(self):
        """Stores all elements making a free space (that is, None) at idx 0."""
        for dst in reversed(range(CFG['history_length'])):  # optimization possibility: circular indices
            src = dst - 1
            self.history[dst] = self.history[src]
        self.history[0] = None


        # # was the_quality line detected in the last iteration?
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

