import numpy as np
import cv2 as cv


def prepare_output(original_frame, frame, separator_size: int = 40):
    if len(frame.shape) == 3:
        h, _, c = frame.shape
    else:
        h, _ = frame.shape
        frame = cv.merge([frame, frame, frame])
        c = 3
    separator = np.zeros((h, separator_size, c), dtype=np.uint8)
    return np.hstack((original_frame, separator, frame))


def convert_kp2np(kps):
    coords = np.float32([keypoint.pt for keypoint in kps])
    return coords.reshape(-1, 1, 2)


def draw_points(pts, frame):
    int_pts = pts.astype(int)
    for i, corner in enumerate(int_pts):
        x, y = corner.ravel()
        color = np.float64([i, 2 * i, 255 - i])
        cv.circle(frame, (x, y), 6, color, 3)
    return frame


def init_kalman():
    kalman = cv.KalmanFilter(6, 2)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32
    )
    kalman.transitionMatrix = np.array(
        [
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        np.float32,
    )
    kalman.processNoiseCov = (
        np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )
        * 1
    )
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 5

    return kalman
