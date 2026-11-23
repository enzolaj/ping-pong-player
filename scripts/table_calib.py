"""Table setup for no-ROS block test: where things are, camera, tags, ball, and Pi.
"""
import argparse
import json
import math
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np


# measured setup

# tags: AprilTag 25h9, 60 mm
TAG_SIZE = 0.060
TAG_DICT = cv2.aruco.DICT_APRILTAG_25h9
TAG_POS = {0: (0.0000, 0.0000), 1: (0.0000, 0.5904), 2: (0.3466, 0.5846), 3: (0.3251, 0.0009)}

# where robot sits: base rotation axis in table coords, and how far model frame is turned
# from table frame (about -90, so model -x = table +y).
ROBOT_BASE_XY = (0.1740, -0.1368)
ROBOT_ROTATION_DEG = -90.977

# 0.025 low enough to stop 40 mm ball
Z_FLOOR = 0.020
Z_BLOCK = 0.025

# values for green funrobo ball
BALL_HSV_LO = (60, 30, 110)
BALL_HSV_HI = (95, 255, 255)

CALIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'calib.json')


# switching between frames
# table = R(ROBOT_ROTATION_DEG) * model + ROBOT_BASE_XY

def model_to_table(model_x, model_y):
    """Model frame point -> table point: rotate by robot's angle, then shift to its base."""
    theta = math.radians(ROBOT_ROTATION_DEG)
    table_x = ROBOT_BASE_XY[0] + math.cos(theta) * model_x - math.sin(theta) * model_y
    table_y = ROBOT_BASE_XY[1] + math.sin(theta) * model_x + math.cos(theta) * model_y
    return table_x, table_y


def table_to_model(table_x, table_y):
    """Undoes model_to_table: take off shift, then rotate back."""
    theta = math.radians(ROBOT_ROTATION_DEG)
    dx = table_x - ROBOT_BASE_XY[0]
    dy = table_y - ROBOT_BASE_XY[1]
    return math.cos(theta) * dx + math.sin(theta) * dy, -math.sin(theta) * dx + math.cos(theta) * dy


def image_to_table(H, u, v):
    """Pixel -> point on floor.

    Homographies work in homogeneous coords, so divide by third value at the end.
    """
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def table_to_image(H, table_x, table_y):
    """Point on floor -> whole pixel, for drawing."""
    # same as image_to_table but homography flipped around
    p = np.linalg.inv(H) @ np.array([table_x, table_y, 1.0])
    return int(p[0] / p[2]), int(p[1] / p[2])


def model_to_image(H, model_x, model_y):
    """Model frame point -> pixel, for drawing where arm is aiming."""
    return table_to_image(H, *model_to_table(model_x, model_y))


def load_homography():
    """Image -> table homography saved by running this file."""
    with open(CALIB_PATH) as f:
        return np.array(json.load(f)['H'])


# camera

class Camera:
    """Reads frames on background thread, only keeps newest one.

    Wifi stream (droidcam) queues frames up, so reading them in order would make program
    fall further and further behind real ball.
    """

    def __init__(self, src):
        # plain number is webcam index, anything else is stream url
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.t = 0.0
        self.count = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        # wait for first frame so read() always has something to give
        while self.frame is None:
            time.sleep(0.01)

    def _run(self):
        # runs on thread, each new frame just overwrites last one
        while self.running:
            ok, frame = self.cap.read()
            t = time.monotonic()      # arrival time, droidcam doesn't send capture times
            if ok:
                with self.lock:
                    self.frame = frame
                    self.t = t
                    self.count += 1

    def read(self):
        """Returns (frame, arrival_time, frame_number).

        frame_number lets caller tell new frame from one it already saw.
        """
        with self.lock:
            return self.frame, self.t, self.count

    def close(self):
        # releasing while thread still inside cap.read() makes opencv's ffmpeg reader crash
        # whole program, so stop thread first
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


# tags

# made once, no point building new detector every frame
_tag_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_DICT),
                                        cv2.aruco.DetectorParameters())


def detect_tags(frame):
    """Finds every tag in view. Returns {printed_id: 4x2 array of corner pixels}."""
    # detector works on grayscale, gives back ids as column array (or None)
    corners, ids, _ = _tag_detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    found = {}
    if ids is not None:
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            found[int(tag_id)] = tag_corners[0]
    return found


def homography_from_tags(found, ids):
    """Image -> table homography from 4 tag centers, or None if one missing.

    ids[position] = printed id at that position. Uses centers not corners bc distances
    were measured center to center.
    """
    image_pts = []
    table_pts = []
    for pos in range(4):
        if ids[pos] not in found:
            return None
        image_pts.append(found[ids[pos]].mean(axis=0))
        table_pts.append(TAG_POS[pos])
    H, _ = cv2.findHomography(np.array(image_pts, np.float32), np.array(table_pts, np.float32))
    return H


def live_homography(cam, ids, n_frames, win):
    """Averages homography over n_frames to smooth out corner jitter.

    Keeps going until n_frames frames have shown all 4 tags, shows what it sees so you can
    fix view while it waits. Returns (H, last_tags_seen).
    """
    homographies = []
    found = {}
    while len(homographies) < n_frames:
        frame, _, _ = cam.read()
        found = detect_tags(frame)
        H = homography_from_tags(found, ids)
        if H is not None:
            # homography only matters up to scale, normalizing lets you average them
            homographies.append(H / H[2, 2])

        vis = frame.copy()
        draw_tags(vis, found)
        draw_text(vis, f'tags seen: {sorted(found)}  need {ids}  ({len(homographies)}/{n_frames})')
        cv2.imshow(win, vis)
        cv2.waitKey(30)

    return sum(homographies) / len(homographies), found


# ball

def detect_ball(frame, lo=BALL_HSV_LO, hi=BALL_HSV_HI, min_area=40, max_area=20000, exclude=None):
    """Biggest roundish blob in ball's color range: (u, v, radius_px), or None.

    `exclude` = optional mask of pixels to ignore (robot, in block test).
    """
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lo, hi)
    if exclude is not None:
        mask[exclude > 0] = 0
    # gets rid of single pixel specks from floor texture before counting blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n_blobs, _, stats, centers = cv2.connectedComponentsWithStats(mask)

    best = None
    best_area = 0
    for i in range(1, n_blobs):     # blob 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        # motion blurred ball stretches out, but not into long strip like cable or edge
        roundish = 0.4 < width / height < 2.5
        if min_area <= area <= max_area and roundish and area > best_area:
            best = i
            best_area = area

    if best is None:
        return None
    radius = 0.5 * max(stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT])
    return centers[best][0], centers[best][1], radius


# drawing

def draw_tags(img, found):
    """Outlines each detected tag, labels it with its printed id."""
    for tag_id, corners in found.items():
        cv2.polylines(img, [corners.astype(np.int32)], True, (0, 200, 255), 2)
        center = tuple(corners.mean(axis=0).astype(int))
        cv2.putText(img, f'id{tag_id}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


def draw_text(img, text, scale=1.0):
    """Status line in the top left corner."""
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)


# talking to pi

class PiLink:
    """UDP to pi_server.py: sends (x, y, z) target in model frame as '<3f'.

    Pi acks each target it runs with '<3f' (IK time, Pi total time, move duration).
    Non-blocking so lost packet never stalls camera loop.
    """

    def __init__(self, ip, port=5005):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def send(self, x, y, z):
        # 3 little-endian floats, format pi_server.py unpacks
        self.sock.sendto(struct.pack('<3f', x, y, z), self.addr)

    def poll_ack(self):
        """Newest ack if one came in, else None (never waits)."""
        try:
            data = self.sock.recv(64)
        except BlockingIOError:
            return None
        return struct.unpack('<3f', data)


# saving tag view

def main():
    """Finds 4 tags, averages homography, saves it to data/calib.json."""
    p = argparse.ArgumentParser()
    p.add_argument('--cam', required=True, help='camera index or droidcam url http://PHONE_IP:4747/video')
    p.add_argument('--ids', nargs=4, type=int, default=[0, 1, 2, 3],
                   help='printed tag ids at positions 0 bottom-left, 1 top-left, 2 top-right, 3 bottom-right')
    args = p.parse_args()

    cam = Camera(args.cam)
    win = 'tags'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print('looking for 4 tags ...')
    H, found = live_homography(cam, args.ids, 30, win)
    for pos in range(4):
        center = found[args.ids[pos]].mean(axis=0)
        print(f'  position {pos} = printed id {args.ids[pos]} at pixel ({center[0]:.0f}, {center[1]:.0f})')

    # saved so block_test.py doesn't need all 4 tags visible (robot covers some)
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, 'w') as f:
        json.dump({'H': H.tolist()}, f, indent=2)
    print(f'saved {os.path.normpath(CALIB_PATH)}')

    cam.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
