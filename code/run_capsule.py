import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import os
import json
from scipy import ndimage, signal
from pathlib import Path
from sklearn.linear_model import HuberRegressor as Regressor
import imageio as io


def subsample_and_crop_video(data_pointer, subsample, crop, start_frame=0, end_frame=-1):
    """Subsample and crop a video, cache results. Also functions as a data_pointer load.

    Args:
        subsample:  An integer specifying the amount of subsampling (1 = full movie)
        crop:  A tuple (px_y, px_x) specifying the number of pixels to remove
        start_frame:  The index of the first desired frame
        end_frame:  The index of the last desired frame

    Returns:
        The resultant array.
    """

    if not isinstance(data_pointer, (h5py._hl.dataset.Dataset, np.ndarray)):
        cropped_video = _subsample_and_crop_video_cv(
            subsample, crop, start_frame=start_frame, end_frame=end_frame
        )
    else:
        _shape = data_pointer.shape
        px_y_start, px_x_start = crop
        px_y_end = _shape[1] - px_y_start
        px_x_end = _shape[2] - px_x_start

        if start_frame == _shape[0] - 1 and (end_frame == -1 or end_frame == _shape[0]):
            cropped_video = data_pointer[
                start_frame::subsample, px_y_start:px_y_end, px_x_start:px_x_end
            ]
        else:
            cropped_video = data_pointer[
                start_frame:end_frame:subsample, px_y_start:px_y_end, px_x_start:px_x_end
            ]

    return cropped_video

def _longest_run(bool_array):
    """
    Find the longest contiguous segment of True values inside bool_array.
    Args:
        bool_array: 1d boolean array.
    Returns:
        Slice with start and stop for the longest contiguous block of True values.
    """
    step = np.diff(np.int8(bool_array), prepend=0, append=0)
    on = np.where(step == 1)[0]
    off = np.where(step == -1)[0]
    i = np.argmax(off - on)
    return slice(on[i], off[i])


def compute_sensitivity(movie: np.array, count_weight_gamma: float=0.2):
    """Calculate photon sensitivity

    Args:
        movie (np.array):  A movie in the format (height, width, time).
        count_weight_gamma: 0.00001=weigh each intensity level equally, 
            1.0=weigh each intensity in proportion to pixel counts.

    Returns:
        dict: A dictionary with the following keys:
            - 'model': The fitted TheilSenRegressor model.
            - 'min_intensity': Minimum intensity used.
            - 'max_intensity': Maximum intensity used.
            - 'variance': Variances at intensity levels.
            - 'sensitivity': Sensitivity.
            - 'zero_level': X-intercept.
    """
    assert (
        movie.ndim == 3
    ), f"A three dimensional (Height, Width, Time) grayscale movie is expected, got {movie.ndim}"

    movie = np.maximum(0, movie.astype(np.int32, copy=False))
    intensity = (movie[:, :, :-1] + movie[:, :, 1:] + 1) // 2
    difference = movie[:, :, :-1].astype(np.float32) - movie[:, :, 1:]

    select = intensity > 0
    intensity = intensity[select]
    difference = difference[select]

    counts = np.bincount(intensity.flatten())
    bins = _longest_run(counts > 0.01 * counts.mean())  # consider only bins with at least 1% of mean counts 
    bins = slice(max(bins.stop * 3 // 100, bins.start), bins.stop)
    assert (
        bins.stop - bins.start > 100
    ), f"The image does not have a sufficient range of intensities to compute the noise transfer function."

    counts = counts[bins]
    idx = (intensity >= bins.start) & (intensity < bins.stop)
    variance = (
        np.bincount(
            intensity[idx] - bins.start,
            weights=(difference[idx] ** 2) / 2,
        )
        / counts
    )
    model = Regressor()
    model.fit(np.c_[bins], variance, counts ** count_weight_gamma)
    sensitivity = model.coef_[0]
    zero_level = - model.intercept_ / model.coef_[0]

    return dict(
        model=model,
        counts=counts,
        min_intensity=bins.start,
        max_intensity=bins.stop,
        variance=variance,
        sensitivity=sensitivity,
        zero_level=zero_level,
    )

def make_luts(zero_level: int, sensitivity: float, input_max: int, beta: float=0.5):
	"""
	Compute lookup tables LUT1 and LUT2.
	LUT1 converts a linear grayscale image into a uniform variance image. 
	LUT2 is the inverse of LUT1 
	:param zero_level: the input level correspoding to zero photon rate
	:param sensitivity: the size of one photon in the linear input image.
	:param beta: the grayscale quantization step in units of noise std dev
	"""
	# produce anscombe LUT1 and LUT2
	xx = (np.r_[:input_max + 1] - zero_level) / sensitivity
	zero_slope = 1 / beta / np.sqrt(3/8)
	offset = -xx.min() * zero_slope
	LUT1 = np.uint8(
		offset +
		(xx < 0) * (xx * zero_slope) +
		(xx >= 0) * (2.0 / beta * (np.sqrt(np.maximum(0, xx) + 3/8) - np.sqrt(3/8))))
	_, LUT2 = np.unique(LUT1, return_index=True)
	LUT2 += (np.r_[:LUT2.size] / LUT2.size * (LUT2[-1] - LUT2[-2])/2).astype('int16')
	return LUT1, LUT2.astype('int16')


def lookup(movie, LUT):
    """
    Apply lookup table LUT to input movie
    """
    return LUT[np.maximum(0, np.minimum(movie, LUT.size-1))]


def save_movie(movie, path, scale, format='gif'):
    if format == "gif":
        with io.get_writer(path, mode='I', duration=.01, loop=False) as f:
            for frame in movie:
                f.append_data(scale * frame)
    else:
        raise NotImplementedError(f"Format {format} is not implemented")


def make_output_directory(output_dir: str, experiment_id: str = None) -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: str
        output directory
    experiment_id: str
        experiment_id number

    Returns
    -------
    output_dir: str
        output directory
    """
    if experiment_id:
        output_dir = os.path.join(output_dir, experiment_id)
    else:
        output_dir = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":  
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Raw movie compression")

    parser.add_argument(
        "-i", "--input-dir", type=str, help="Parent directory of raw movie", default="../data/"
    )

    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory", default="/results/"
    )

    parser.add_argument(
        "--crop",
        type=list,
        default=(30, 30),
        help=("cropped area of movie to use for analysis. Useful to remove" " edge effects"),
    )

    # This is to constrain the analysis to a subset of frames
    # to reduce cost and increase speed
    parser.add_argument(
        "--start_frame", type=int, default=1, help=("Start of movie block to use for main analysis")
    )

    parser.add_argument(
        "--end_frame", type=int, default=1000, help=("End of movie block to use for main analysis")
    )

    # Parse command-line arguments
    args = parser.parse_args()
    # General settings

    # name of the dataset in the hdf5 file
    dataset_name = "data"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    h5_file = [i for i in list(input_dir.glob("*/*")) if "output.h5" in str(i)][0]
    experiment_id = h5_file.name.split("_")[0]
    output_dir = make_output_directory(output_dir, experiment_id)
    # processing_json_fp = h5_file.parent / "processing.json"

    # with open(processing_json_fp, "r") as j:
    #    data = json.load(j)

    frame_rate = 10 # data["data_processes"][0]["parameters"]["movie_frame_rate_hz"]
    
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
      
    print(cropped_video.shape)

    qs = compute_sensitivity(cropped_video.transpose(1, 2, 0), count_weight_gamma=1)
    print('Quantal size: {sensitivity:5.1f}\nIntercept: {zero_level:5.1f}\n'.format(f=**qs))

    metrics = {}

    metrics['sensitivity'] = qs['sensitivity']
    metrics['counts'] = qs['zero_level']
    metrics['counts'] = qs['counts']

    print(metrics)

    fig = plt.figure(figsize=(1.8, 2.6))
    gs = fig.add_gridspec(
        1, 1,
        left=0.0, right=1.0, bottom=0.0, top=0.9)

    ax = fig.add_subplot(gs[0])

    matplotlib.rc('font', family='sans', size=8)

    m = cropped_video.mean(axis=0)
    _ = ax.imshow(m, vmin=0, vmax=np.quantile(m, 0.999), cmap='gray')
    ax.axis(False)
    ax.set_title('mean fluorescence')
    ax.title.set_size(8)
    fig.savefig(output_dir.parent / 'A.png', dpi=300)