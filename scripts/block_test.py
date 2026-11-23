"""No-ROS block test: overhead phone camera -> ball track -> predicted crossing -> arm target.

Arm waits on line `--fwd` m in front of its base. Every frame, ball position goes
camera -> table (saved tag homography) -> model frame (robot pose) into trajectory fit, and
gripper gets sent to where fit says ball will cross line. Pi does IK.

    /usr/bin/python3 scripts/block_test.py --cam http://PHONE_IP:4747/video

Keys: q quit | r back to ready pose | m show what ball detector sees
Every run logged to data/block_runs/.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pingpong.trajectory_fit import TrajectoryFitter, predict_crossing  # noqa: E402
import table_calib as tc  # noqa: E402


# settings

RESEND_DIST = 0.01      # m, smaller corrections not worth interrupting move for
RESEND_GAP = 0.08       # s, pi only takes newest target after each move anyway
BALL_GONE_S = 1.5       # s without seeing ball before arm goes back to ready

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'block_runs')
LOG_FIELDS = ['t', 'u_px', 'v_px', 'X', 'Y', 'xm', 'ym', 'fit_n', 'fit_residual_mm',
              'time_to_cross_s', 'y_hit', 'sent_y', 'event']


def parse_args():
    """Command line options. Everything but --cam defaults to what worked on real setup."""
    p = argparse.ArgumentParser()
    p.add_argument('--cam', required=True, help='camera index or droidcam url http://PHONE_IP:4747/video')
    p.add_argument('--pi', default='192.168.16.64')
    p.add_argument('--fwd', type=float, default=0.18,
                   help='block line distance in front of the base axis (m), chassis front is ~0.12')
    p.add_argument('--z', type=float, default=tc.Z_BLOCK,
                   help='gripper height, model z (floor is 0.020)')
    p.add_argument('--ymax', type=float, default=0.15,
                   help='max sideways reach along the line (m), IK reaches +-0.18 at fwd 0.18')
    p.add_argument('--min-points', type=int, default=6, help='detections before trusting a prediction')
    p.add_argument('--hsv', nargs=6, type=int, default=tc.BALL_HSV_LO + tc.BALL_HSV_HI,
                   help='ball hsv lo/hi: h s v h s v')
    return p.parse_args()


# setup helpers

def robot_zone(cam, H, fwd):
    """Pixel polygon and mask of everything between block line and robot.

    Robot's blue parts land in ball's color range, and ball in there already reached
    line anyway, so ignoring it loses nothing.
    """
    # corners as (distance in front of base, sideways): just short of line back to well
    # behind robot, wider than camera can see
    corners = [(fwd - 0.005, -0.35), (fwd - 0.005, 0.35), (-0.30, 0.35), (-0.30, -0.35)]
    poly = []
    for ahead, side in corners:
        poly.append(tc.model_to_image(H, -ahead, side))
    poly = np.array(poly, np.int32)

    frame, _, _ = cam.read()
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return poly, mask


def open_log():
    """New csv in data/block_runs/ named after start time. Returns (writer, path)."""
    # DictWriter so rows fill in by column name, missing ones left blank
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, time.strftime('run_%Y%m%d_%H%M%S.csv'))
    log = csv.DictWriter(open(path, 'w', newline=''), fieldnames=LOG_FIELDS, restval='')
    log.writeheader()
    return log, path


# arm and prediction

class Arm:
    """Remembers last target so loop only resends when gripper would actually move."""

    def __init__(self, pi, ready):
        self.pi = pi
        self.ready = ready
        self.go(ready, time.monotonic())

    def go(self, target, now):
        """Sends (x, y, z) model frame target, remembers what got sent and when."""
        self.pi.send(*target)
        self.target = target
        self.t_sent = now

    def at_ready(self):
        # so arm only gets sent home once after ball disappears
        return self.target == self.ready


def predict(fitter, fwd, min_points):
    """(t_cross, y_hit) for current fit, or None if nothing worth trusting."""
    # too few points = fit swings around a lot, so wait for min_points
    fit = fitter.last_fit
    if fit is None or fit['n'] < min_points:
        return None
    t_cross, y_hit = predict_crossing(fit, fwd)
    if t_cross is None:
        return None
    return t_cross, y_hit


# main loop

def main():
    """Sets everything up, then tracks ball and moves arm every frame until q."""
    args = parse_args()
    hsv_lo, hsv_hi = tuple(args.hsv[:3]), tuple(args.hsv[3:])

    # camera, plus overlays that only depend on where camera is (it doesn't move)
    cam = tc.Camera(args.cam)
    win = 'block test'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    H = tc.load_homography()     # saved by table_calib.py, rerun that if camera moved
    zone_px, zone_mask = robot_zone(cam, H, args.fwd)
    line_px = [tc.model_to_image(H, -args.fwd, -args.ymax),
               tc.model_to_image(H, -args.fwd, args.ymax)]

    # fitter's x = distance in front of robot (-model x), so ball rolls toward -x and
    # crosses at x = fwd, what predict_crossing expects
    fitter = TrajectoryFitter(window_s=0.6, min_points=5, accel_min_points=15, max_gap=0.15, outlier_dist=0.04)
    ready = (-args.fwd, 0.0, args.z)
    arm = Arm(tc.PiLink(args.pi), ready)

    log, log_path = open_log()
    print(f'ready pose sent: model {ready}; logging to {os.path.normpath(log_path)}')

    last_frame_number = -1
    last_seen = 0.0
    show_mask = False
    try:
        while True:
            # grab frame we haven't done yet, camera thread can be slower than us
            frame, t, frame_number = cam.read()
            if frame_number == last_frame_number:
                cv2.waitKey(2)
                continue
            last_frame_number = frame_number
            now = time.monotonic()
            row = {'t': round(t, 4), 'event': ''}

            vis = frame.copy()
            cv2.line(vis, line_px[0], line_px[1], (255, 0, 255), 2)
            cv2.polylines(vis, [zone_px], True, (128, 128, 128), 1)

            ball = tc.detect_ball(frame, hsv_lo, hsv_hi, exclude=zone_mask)
            if ball is not None:
                # ball seen: pixels -> table -> model frame, then into fit
                u, v, radius_px = ball
                table_x, table_y = tc.image_to_table(H, u, v)
                model_x, model_y = tc.table_to_model(table_x, table_y)
                if fitter.update(t, -model_x, model_y) == 'reset':
                    row['event'] = 'fit reset'
                last_seen = now

                cv2.circle(vis, (int(u), int(v)), int(radius_px) + 4, (0, 255, 0), 2)
                row.update(u_px=round(u, 1), v_px=round(v, 1), X=round(table_x, 4), Y=round(table_y, 4),
                           xm=round(model_x, 4), ym=round(model_y, 4))
                fit = fitter.last_fit
                if fit is not None:
                    row.update(fit_n=fit['n'], fit_residual_mm=round(1000 * fit['residual'], 2))

                # predict crossing and send gripper there, kept inside its reach
                crossing = predict(fitter, args.fwd, args.min_points)
                if crossing is not None:
                    t_cross, y_hit = crossing
                    y_cmd = min(max(y_hit, -args.ymax), args.ymax)
                    row.update(time_to_cross_s=round(t_cross - t, 3), y_hit=round(y_hit, 4))
                    cv2.circle(vis, tc.model_to_image(H, -args.fwd, y_cmd), 10, (0, 0, 255), 3)

                    moved_enough = abs(y_cmd - arm.target[1]) > RESEND_DIST
                    waited_enough = now - arm.t_sent > RESEND_GAP
                    if moved_enough and waited_enough:
                        arm.go((-args.fwd, y_cmd, args.z), now)
                        row.update(sent_y=round(y_cmd, 4), event='sent')
                        print(f'send y = {y_cmd:+.3f} m | crossing in {t_cross - t:.2f} s | '
                              f'n = {fit["n"]}, residual {1000 * fit["residual"]:.1f} mm')

            elif now - last_seen > BALL_GONE_S and not arm.at_ready():
                # ball gone a while so roll's over, get ready for next one
                fitter.reset()
                arm.go(ready, now)
                row['event'] = 'ready'
            log.writerow(row)

            ack = arm.pi.poll_ack()
            if ack:
                print(f'  ack: IK {1000 * ack[0]:.0f} ms, Pi {1000 * ack[1]:.0f} ms, move {ack[2]:.2f} s')

            # show camera view, or with 'm' detector's color mask (for tuning hsv)
            if show_mask:
                mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), hsv_lo, hsv_hi)
                mask[zone_mask > 0] = 0
                vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            n_fit = fitter.last_fit['n'] if fitter.last_fit else 0
            tc.draw_text(vis, f'cmd y {arm.target[1]:+.3f}  fit n {n_fit}')
            cv2.imshow(win, vis)

            # keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                fitter.reset()
                arm.go(ready, now)
            if key == ord('m'):
                show_mask = not show_mask
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
