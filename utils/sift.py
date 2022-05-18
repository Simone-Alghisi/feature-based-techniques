from argparse import _SubParsersAction as SubParser
import cv2 as cv


def configure_subparsers(subparsers: SubParser):
    parser = subparsers.add_parser("sift", help="Initialise the SIFT feature detector")
    parser.add_argument(
        "--n-features",
        "-nf",
        metavar="N_FEATURES",
        type=int,
        default=100,
        help="The number of best features to retain",
    )
    parser.add_argument(
        "--n-layers",
        "-nl",
        metavar="N_OCTAVE_ALYERS",
        type=int,
        default=3,
        help="The number of layers in each octave",
    )
    parser.add_argument(
        "--sigma",
        "-s",
        metavar="SIGMA",
        type=float,
        default=1.6,
        help="Sigma value to apply at the first layer in each octave",
    )

    parser.set_defaults(func=main)
    parser.set_defaults(name="sift")


def main(args, frame):
    sift = cv.SIFT_create(
        nfeatures=args.n_features, nOctaveLayers=args.n_layers, sigma=args.sigma
    )
    kp, des = sift.detectAndCompute(frame, None)
    return kp, des
