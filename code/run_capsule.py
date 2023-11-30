import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import os
import json
from scipy import ndimage, signal
from pathlib import Path


if __name__ == "__main__":  
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Raw movie compression")

    parser.add_argument(
        "-i", "--input-dir", type=str, help="Parent directory of raw movie", default="../data/"
    )

    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory", default="/results/"
    )

    # This is to constrain the analysis to a subset of frames
    # to reduce cost and increase speed
    parser.add_argument(
        "--start_frame", type=int, default=1, help=("Start of movie block to use for main analysis")
    )

    parser.add_argument(
        "--end_frame", type=int, default=10000, help=("End of movie block to use for main analysis")
    )

    # Parse command-line arguments
    args = parser.parse_args()
    # General settings

    # name of the dataset in the hdf5 file
    dataset_name = "data"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    h5_file = [i for i in list(input_dir.glob("*/*")) if "registered.h5" in str(i)][0]
    experiment_id = h5_file.name.split("_")[0]
    output_dir = make_output_directory(output_dir, experiment_id)
    processing_json_fp = h5_file.parent / "processing.json"

    with open(processing_json_fp, "r") as j:
        data = json.load(j)

    frame_rate = data["data_processes"][0]["parameters"]["movie_frame_rate_hz"]
    
    with h5py.File(h5_file, "r") as h5_pointer:
        data_pointer = h5_pointer[dataset_name]

        # in principle, this is done once to avoid loading the data multiple times
        # We generate a few smaller version of the big movie to use for metrics
        cropped_video = subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=1,
            crop=args.crop,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
      
    metrics = {}
