from argparse import _SubParsersAction as SubParser
import cv2 as cv
import numpy as np


def configure_subparsers(subparsers: SubParser):
    parser = subparsers.add_parser(
        "gftt", help="Initialise the Good-Feature-To-Track corner detector"
    )
    parser.add_argument(
        "--max-corners",
        "-mc",
        metavar="MAX_CORNERS",
        type=int,
        default=100,
        help="Returns at most MAX_CORNERS",
    )
    parser.add_argument(
        "--quality",
        "-q",
        metavar="QUALITY",
        type=float,
        default=0.01,
        help="Threshold for how good the corners should be",
    )
    parser.add_argument(
        "--min-distance",
        "-md",
        metavar="MIN_DISTANCE",
        type=float,
        default=10,
        help="Minimum possible euclidean distance between the corners",
    )
    parser.add_argument(
        "--block-size",
        "-bs",
        metavar="BLOCK_SIZE",
        type=int,
        default=3,
        help="size of an average block for computing a derivative matrix",
    )
    parser.add_argument(
        "--harris",
        metavar="K_HARRIS",
        type=float,
        default=None,
        help="If set, uses the Harris detector for the corners. It also needs the K parameter",
    )
    parser.set_defaults(func=main)
    parser.set_defaults(name="gftt")


def main(args, frame):
    corners = cv.goodFeaturesToTrack(
        frame,
        maxCorners=args.max_corners,
        qualityLevel=args.quality,
        minDistance=args.min_distance,
        blockSize=args.block_size,
        useHarrisDetector=True if args.harris else False,
        k=args.harris,
    )

    return corners
