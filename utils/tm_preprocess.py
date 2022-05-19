from argparse import _SubParsersAction as SubParser
from typing import Tuple
import cv2 as cv
import numpy as np
from datetime import datetime


def configure_subparsers(subparsers: SubParser):
    parser = subparsers.add_parser(
        "tm_preprocess",
        help="Initialise and Preprocess Template and Frame for Template Matching",
    )
    parser.add_argument(
        "--template",
        "-t",
        metavar="TEMPLATE",
        type=str,
        default=None,
        help="Path to the template [default: None]",
    )
    parser.add_argument(
        "--save",
        "-s",
        default=False,
        action="store_true",
        help="Save the template in the 'templates' folder [default: False]",
    )
    parser.add_argument(
        "--threshold",
        "-th",
        metavar="THRESHOLD",
        type=float,
        default=0.8,
        help="Threshold for the template matcher [default: 0.8]",
    )
    parser.add_argument(
        "--gaussian-blur",
        "-gb",
        default=False,
        action="store_true",
        help="Apply Gaussian Blur to remove edges [default: False]",
    )
    parser.add_argument(
        "--binarize",
        "-b",
        metavar=("THRESH", "MAX_VALUE"),
        type=int,
        nargs=2,
        help="Apply Inverse Binarisation",
    )
    parser.add_argument(
        "--canny",
        "-c",
        metavar=("LOWER_THRESH", "UPPER_THRESH"),
        type=int,
        nargs=2,
        default=None,
        help="Apply Canny edge extractor to the image [default: None]",
    )
    parser.set_defaults(func=main)
    parser.set_defaults(name="tm_preprocess")


def main(args, frame, frame_idx):
    if args.gaussian_blur:
        frame = cv.GaussianBlur(frame, (5, 5), 0)
    if args.binarize is not None:
        # 80 255
        _, frame = cv.threshold(frame, *args.binarize, cv.THRESH_BINARY_INV)
    if args.canny is not None:
        # 150 200
        frame = cv.Canny(frame, *args.canny)

    if frame_idx == 0:
        if args.template is None:
            x, y, w, h = cv.selectROI(frame, False)
            cv.destroyAllWindows()
            args.template = frame[y : y + h, x : x + w]
        else:
            args.template = cv.imread(args.template, cv.IMREAD_GRAYSCALE)
            if args.gaussian_blur:
                args.template = cv.GaussianBlur(args.template, (5, 5), 0)
            if args.binarize is not None:
                _, args.template = cv.threshold(
                    args.template, *args.binarize, cv.THRESH_BINARY_INV
                )
            if args.canny is not None:
                args.template = cv.Canny(args.template, *args.canny)

            if args.save:
                n = datetime.now()
                cv.imwrite(f"templates/{str(n)}.jpg", args.template)

        cv.imshow("template", args.template)

    return frame
