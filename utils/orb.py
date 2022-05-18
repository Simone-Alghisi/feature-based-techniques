from argparse import _SubParsersAction as SubParser
import cv2 as cv


def configure_subparsers(subparsers: SubParser):
    parser = subparsers.add_parser("orb", help="Initialise ORB feature detector")
    parser.add_argument(
        "--n-features",
        "-nf",
        metavar="N_FEATURES",
        type=int,
        default=100,
        help="The number of best features to retain",
    )
    parser.add_argument(
        "--scale-factor",
        "-sf",
        metavar="SCALE_FACTOR",
        type=float,
        default=1.2,
        help="The pyramid decimation ratio",
    )
    parser.add_argument(
        "--n-levels",
        "-nl",
        metavar="N_LEVELS",
        type=int,
        default=8,
        help="The numer of layers in the pyramid",
    )

    parser.set_defaults(func=main)
    parser.set_defaults(name="orb")


def main(args, frame):
    orb = cv.ORB_create(
        nfeatures=args.n_features, scaleFactor=args.scale_factor, nlevels=args.n_levels
    )
    kp, des = orb.detectAndCompute(frame, None)
    return kp, des
