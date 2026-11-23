"""Fits rolling ball's path on table.

Constant accel on each axis (friction slows it pretty steadily), fit to last fraction of a
second of detections. Times measured from newest detection, so fitted start values = where
ball is right now and how fast it's going.
"""
import math

import numpy as np


# physics

def position_at_constant_accel(start, velocity, accel, dt):
    """Where ball at `start`, moving at `velocity` with constant `accel`, ends up `dt` s later.

    `dt` can be negative (time in the past), that's how fit compares itself to older detections.
    """
    return start + velocity * dt + 0.5 * accel * dt * dt


def fit_axis(times, positions, use_accel):
    """Least squares fit of one axis to position_at_constant_accel.

    `times` relative to newest detection (all <= 0). With use_accel off, accel stuck at 0,
    so just straight constant-speed fit.
    Returns (position, velocity, accel, sum_sq_error), position/velocity at time 0.
    """
    # rows of [1, dt, dt^2/2] so solved coefficients come out as position, velocity, accel
    rows = []
    for dt in times:
        rows.append([1.0, dt, 0.5 * dt * dt] if use_accel else [1.0, dt])
    coeffs = np.linalg.lstsq(np.array(rows), np.array(positions), rcond=None)[0]
    position = coeffs[0]
    velocity = coeffs[1]
    accel = coeffs[2] if use_accel else 0.0

    sum_sq_error = 0.0
    for dt, measured in zip(times, positions):
        sum_sq_error += (measured - position_at_constant_accel(position, velocity, accel, dt)) ** 2
    return position, velocity, accel, sum_sq_error


# using fits

def predicted_xy(fit, t):
    """Where fit says ball is (or was) at absolute time t."""
    dt = t - fit['t0']
    x = position_at_constant_accel(fit['x0'], fit['vx'], fit['ax'], dt)
    y = position_at_constant_accel(fit['y0'], fit['vy'], fit['ay'], dt)
    return x, y


def predict_crossing(fit, x_line):
    """When and where ball hits x = x_line while rolling toward -x.

    Returns (t_cross, y_hit), or (None, None) if it never gets there.
    """
    gap = fit['x0'] - x_line          # how far left to roll
    velocity = fit['vx']
    accel = fit['ax']
    if gap <= 0.0 or velocity >= 0.0:
        return None, None              # already past line, or rolling away

    # solve x(dt) = x_line, i.e. (accel/2) dt^2 + velocity dt + gap = 0
    if abs(accel) < 0.0001:
        dt = -gap / velocity           # basically no accel, so just distance / speed
    else:
        discriminant = velocity * velocity - 2.0 * accel * gap
        if discriminant < 0.0:
            return None, None          # friction stops it before line
        future_roots = []
        for root in ((-velocity - math.sqrt(discriminant)) / accel,
                     (-velocity + math.sqrt(discriminant)) / accel):
            if root > 0.0:
                future_roots.append(root)
        if not future_roots:
            return None, None
        dt = min(future_roots)

    # parabola keeps going after ball would've stopped, like it rolled back. root on
    # that part not a real crossing
    if velocity + accel * dt > 0.0:
        return None, None
    t_cross = fit['t0'] + dt
    return t_cross, predicted_xy(fit, t_cross)[1]


# live fitter

class TrajectoryFitter:
    """Keeps fitting over sliding window of detections, call update() for every detection.

    window_s          only detections from last window_s seconds get fit
    max_points        hard cap on detections in fit
    min_points        detections needed before any fit
    accel_min_points  detections needed before fitting accel too
    max_gap           s without detection that counts as new roll
    outlier_dist      m away from prediction before detection looks sus
    max_outliers      sus detections in a row before reset
    """

    def __init__(self, window_s=0.5, max_points=100, min_points=5, accel_min_points=15,
                 max_gap=0.08, outlier_dist=0.03, max_outliers=3):
        self.window_s = window_s
        self.max_points = max_points
        self.min_points = min_points
        # stays constant-velocity until enough points, bc accel from few noisy points
        # mostly noise
        self.accel_min_points = accel_min_points
        self.max_gap = max_gap
        self.outlier_dist = outlier_dist
        self.max_outliers = max_outliers
        self.reset()

    def reset(self):
        # forget current roll, next detection starts fresh track
        self.times = []
        self.xs = []
        self.ys = []
        self.outliers_in_a_row = 0
        self.last_fit = None

    def update(self, t, x, y):
        """Add one detection. Returns 'ok', 'outlier' or 'reset'."""
        status = 'ok'
        if self.times and t - self.times[-1] > self.max_gap:
            # long gap = lost track or new roll started, old points would just mislead
            self.reset()
            status = 'reset'
        elif self.last_fit is not None and self._far_from_prediction(t, x, y):
            # skip few stray detections, but if they keep coming ball actually changed
            # course (like it hit something) and old fit wrong
            self.outliers_in_a_row += 1
            if self.outliers_in_a_row <= self.max_outliers:
                return 'outlier'
            self.reset()
            status = 'reset'

        self.outliers_in_a_row = 0
        self._add(t, x, y)
        self.last_fit = self.fit()
        return status

    def _far_from_prediction(self, t, x, y):
        # compare new detection to where last fit said ball would be by now
        predicted_x, predicted_y = predicted_xy(self.last_fit, t)
        return math.hypot(x - predicted_x, y - predicted_y) > self.outlier_dist

    def _add(self, t, x, y):
        self.times.append(t)
        self.xs.append(x)
        self.ys.append(y)
        # only keep recent past, friction and spin slowly change motion so old points
        # drag fit off
        while len(self.times) > self.max_points or t - self.times[0] > self.window_s:
            self.times.pop(0)
            self.xs.pop(0)
            self.ys.pop(0)

    def fit(self):
        """Fits current window, or None if not enough points yet.

        Returns dict: t0 (time of newest detection), x0/y0 and vx/vy (position and velocity
        at t0), ax/ay (accel), n (detections used), residual (rms distance of detections
        from fit, m).
        """
        n = len(self.times)
        if n < self.min_points:
            return None
        use_accel = n >= self.accel_min_points

        # times measured back from newest detection so x0/vx come out as ball's current state
        t0 = self.times[-1]
        times_from_newest = []
        for t in self.times:
            times_from_newest.append(t - t0)
        x0, vx, ax, x_sq_error = fit_axis(times_from_newest, self.xs, use_accel)
        y0, vy, ay, y_sq_error = fit_axis(times_from_newest, self.ys, use_accel)
        return {'t0': t0, 'x0': x0, 'vx': vx, 'ax': ax, 'y0': y0, 'vy': vy, 'ay': ay,
                'n': n, 'residual': math.sqrt((x_sq_error + y_sq_error) / (2 * n))}
