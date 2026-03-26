import tifffile
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Checks for nonzero values of an annotation TIFF file."
    )
    parser.add_argument("--tiff_path", required=True,
                        help="Path to TIFF file.")
    args = parser.parse_args()
    path = args.tiff_path

    test_label = tifffile.imread(path)
    print(np.unique(test_label))
