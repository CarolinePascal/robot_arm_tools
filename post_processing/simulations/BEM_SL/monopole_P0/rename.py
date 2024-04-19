import os
import glob
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("underscore_pattern", help="Renamed files pattern", type=str)
    parser.add_argument("new_ending_value", help="New value to place after the last underscore", type=str)
    args = parser.parse_args()

    files = glob.glob(args.underscore_pattern)
    for file in files:
        basename,extension = os.path.basename(file),file.split(".")[-1]
        new_file = "_".join(basename.split("_")[:-1] + [args.new_ending_value]) + "." + extension
        os.rename(file, new_file)