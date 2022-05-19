import argparse
from argparse import RawTextHelpFormatter
import cv2 as cv
import numpy as np
from utils import gftt, sift, orb, tm_preprocess
from utils.utils import convert_kp2np, draw_points, init_kalman, prepare_output


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="cv-assignment",
        description="Program written for the Computer Vision assignment",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "algorithm",
        metavar="ALGORITHM",
        type=str,
        choices=["lk", "bfm", "mtm", "k"],
        help="""Algorithm to run. The following are available:
- lk (Lucas-Kanade), i.e use Lucas-Kanade optical flow to track salient points;
- bfm (Bruteforce Matcher), i.e. use a Bruteforce Matcher to track keypoints and descriptors;
- mtm (Multiple Template Matching), i.e. match a given template multiple times across the frame;
- k (Kalman Filter), i.e. predict the next position of some salient points in the scene.
        """,
    )
    parser.add_argument(
        "video",
        metavar="VIDEO",
        type=str,
        help="Path to the video to be processed.",
    )
    parser.add_argument(
        "--max-frames",
        "-mf",
        metavar="MAX_FRAMES",
        default="1000",
        type=int,
        help="Max number of frame to analyse in the video [default: 1000]",
    )
    parser.add_argument(
        "--scale",
        "-s",
        metavar="SCALE",
        type=float,
        help="Size for rescaling the video [default: 0.2]",
        default="0.2",
    )
    parser.add_argument(
        "--sampling-rate",
        "-sr",
        metavar="SAMPLING_RATE",
        type=int,
        help="Sampling rate for updating keypoints [default: 50]",
        default="50",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="VIDEO_NAME",
        type=str,
        help="If specified, saves the video as 'VIDEO_NAME' using the provided format [default: None]",
        default=None,
    )

    # subparsers
    subparsers = parser.add_subparsers(help="sub-commands help")
    gftt.configure_subparsers(subparsers)
    sift.configure_subparsers(subparsers)
    orb.configure_subparsers(subparsers)
    tm_preprocess.configure_subparsers(subparsers)

    parsed_args = parser.parse_args()
    if parsed_args.algorithm in ["lk"]:
        if "func" not in parsed_args or parsed_args.name not in ["gftt", "sift", "orb"]:
            parser.exit(
                1,
                "Cannot use Lucas-Kanade without specifying a corner detector (gftt, sift, orb)",
            )
    elif parsed_args.algorithm in ["bfm"]:
        if "func" not in parsed_args or parsed_args.name not in ["sift", "orb"]:
            parser.exit(
                1,
                "Cannot use a Matcher without specifying a feature detector (sift, orb)",
            )
    elif parsed_args.algorithm in ["mtm"]:
        if "func" not in parsed_args or parsed_args.name not in ["tm_preprocess"]:
            parser.exit(
                1,
                "Cannot use a Template Matching without using tm_preprocess",
            )
    elif parsed_args.algorithm in ["k"]:
        if "func" not in parsed_args or parsed_args.name not in ["gftt", "sift", "orb"]:
            parser.exit(
                1,
                "Cannot use Kalman Filter without specifying a corner detector (gftt, sift, orb)",
            )

    return parsed_args


def main(args):
    cap = cv.VideoCapture(args.video)

    prev_frame = None
    prev_corners = None
    prev_des = None
    kalman = None

    for i in range(args.max_frames):
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

        # Capture frame-by-frame
        ret, frame = cap.read()

        # If video end reached
        if not ret:
            break

        frame = cv.resize(frame, (int(width * args.scale), int(height * args.scale)))
        frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        if args.algorithm == "lk":
            to_display, prev_frame, prev_corners = lk(
                args, frame, frame_gray, i, prev_frame, prev_corners
            )
        elif args.algorithm == "bfm":
            to_display, prev_frame, prev_des = bfm(
                args, frame, frame_gray, i, prev_frame, prev_des
            )
        elif args.algorithm == "mtm":
            to_display = multiple_template_matching(args, frame, frame_gray, i)
        elif args.algorithm == "k":
            if kalman is None:
                kalman = init_kalman()
            to_display, prev_frame, prev_corners = kalman_tr(
                args,
                frame,
                frame_gray,
                i,
                kalman,
                prev_frame,
                prev_corners,
            )

        cv.imshow(*to_display)

        if args.output:
            if i == 0:
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                out = cv.VideoWriter(
                    args.output,
                    fourcc,
                    20.0,
                    (to_display[-1].shape[1], to_display[-1].shape[0]),
                )
            out.write(to_display[-1])

        # Wait and exit if q is pressed
        if cv.waitKey(1) == ord("q") or not ret:
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def lk(args, frame, frame_gray, frame_idx, prev_frame=None, prev_corners=None):
    frame_lk = frame.copy()
    if frame_idx % args.sampling_rate == 0:
        if args.name == "gftt":
            corners = args.func(args, frame_gray)
        else:
            kp, _ = args.func(args, frame_gray)
            corners = convert_kp2np(kp)

    else:
        corners, _, _ = cv.calcOpticalFlowPyrLK(
            prev_frame, frame_gray, prev_corners, None
        )

    frame_lk = draw_points(corners, frame_lk)

    # Copy values for next iteration
    prev_frame, prev_corners = frame_gray.copy(), corners
    to_display = prepare_output(frame, frame_lk)
    return (
        ("lucas-kanade with {}".format(args.name), to_display),
        prev_frame,
        prev_corners,
    )


def bfm(args, frame, frame_gray, frame_idx, prev_frame=None, prev_des=None):
    frame_bfm = frame.copy()
    if frame_idx % args.sampling_rate == 0:
        prev_kp, prev_des = args.func(args, frame_gray)
        prev_kp = convert_kp2np(prev_kp)
        prev_frame = draw_points(prev_kp, frame.copy())

    kp, des = args.func(args, frame_gray)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des, prev_des, k=2)
    good = []
    for match in matches:
        (m, n) = match
        if m.distance < 0.75 * n.distance:
            good.append(kp[m.queryIdx])

    pts = convert_kp2np(good)
    frame_bfm = draw_points(pts, frame_bfm)
    to_display = prepare_output(prev_frame, frame)
    to_display = prepare_output(to_display, frame_bfm)
    return (
        ("BruteForceMatcher with {}".format(args.name), to_display),
        prev_frame,
        prev_des,
    )


def multiple_template_matching(args, frame, frame_gray, frame_idx):
    frame_pp = args.func(args, frame_gray, frame_idx)
    w, h = args.template.shape[::-1]
    res = cv.matchTemplate(frame_pp, args.template, cv.TM_CCOEFF_NORMED)

    frame_tm = frame.copy()
    loc = np.where(res >= args.threshold)
    cv.putText(
        frame_tm,
        text="OBJECTS DETECTED: {}".format(len(loc[0])),
        org=(20, 50),
        fontFace=cv.FONT_HERSHEY_DUPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
    )
    for pt in zip(*loc[::-1]):
        cv.rectangle(frame_tm, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    to_display = prepare_output(frame, frame_pp)
    to_display = prepare_output(to_display, frame_tm)

    return ("Template matching", to_display)


def kalman_tr(
    args,
    frame,
    frame_gray,
    frame_idx,
    kalman,
    prev_frame=None,
    prev_corners=None,
):
    frame_k = frame.copy()
    if frame_idx % args.sampling_rate == 0:
        if args.name == "gftt":
            corners = args.func(args, frame_gray)
        else:
            kp, _ = args.func(args, frame_gray)
            corners = convert_kp2np(kp)

    else:
        corners, _, _ = cv.calcOpticalFlowPyrLK(
            prev_frame, frame_gray, prev_corners, None
        )
    for corner in corners:
        x, y = corner.ravel()
        current_mes = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
        kalman.correct(current_mes)
        current_pre = kalman.predict()

        cmx, cmy = [l.astype(int) for l in current_mes[:2]]
        cpx, cpy = [l.astype(int) for l in current_pre[:2]]
        cv.circle(frame_k, (cmx[0], cmy[0]), 6, np.float64([0, 200, 0]), 3)
        cv.circle(frame_k, (cpx[0], cpy[0]), 6, np.float64([0, 0, 200]), 3)

    to_display = prepare_output(frame, frame_k)
    prev_frame, prev_corners = frame_gray.copy(), corners

    return (
        ("Kalman Filter with {}".format(args.name), to_display),
        prev_frame,
        prev_corners,
    )


if __name__ == "__main__":
    main(get_args())
