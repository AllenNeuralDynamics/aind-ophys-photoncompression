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
import matplotlib
import colorcet as cc
import glob
from sync_dataset import Sync

def get_frame_rate_from_sync(sync_file, platform_data) -> float:
    """Calculate frame rate from sync file
    Parameters
    ----------
    sync_file: str
        path to sync file
    platform_data: dict
        platform data from platform.json
    Returns
    -------
    frame_rate_hz: float
        frame rate in Hz
    """
    labels = ["vsync_2p", "2p_vsync"]  # older versions of sync may 2p_vsync label
    imaging_groups = len(
        platform_data["imaging_plane_groups"]
    )  # Number of imaging plane groups for frequency calculation
    frame_rate_hz = None
    for i in labels:
        sync_data = Sync(sync_file)
        try:
            rising_edges = sync_data.get_rising_edges(i, units="seconds")
            image_freq = 1 / (np.mean(np.diff(rising_edges)))
            frame_rate_hz = image_freq / imaging_groups

        except ValueError:
            pass
    sync_data.close()
    if not frame_rate_hz:
        raise ValueError(f"Frame rate no acquired, line labels: {sync_data.line_labels}")
    return frame_rate_hz

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
        "-i", "--input_searchpath", type=str, help="Regular expression to input hdf5 movie. The first one found is picked", default="../data/*/*/*/*.h5"
    )

    parser.add_argument(
        "--plane_number", type=int, help="index of plane", default=0
    )

    parser.add_argument(
        "--output_dir", type=str, help="Output directory", default="/results/"
    )

    parser.add_argument(
        "--binning",
        type=int,
        default=2,
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
        "--end_frame", type=int, default=999, help=("End of movie block to use for main analysis")
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # General settings
    binning = args.binning

    # name of the dataset in the hdf5 file
    dataset_name = "data"

    plane_number = args.plane_number

    output_dir = Path(args.output_dir)

    list_files = glob.glob(args.input_searchpath)

    # We remove stack files
    list_files = [indiv_path for indiv_path in list_files if not "stack" in indiv_path]

    print("list files found:")
    print(list_files)
    h5_file = Path(list_files[plane_number])

    print("h5 file used:")
    print(h5_file)
    experiment_id = h5_file.name.split("_")[0]
    output_dir = make_output_directory(output_dir, "photoncompression")

    experiment_id = h5_file.name.split("_")[0].split(".")[0]
    print(experiment_id)

    session_folder = h5_file.parent.parent
    platform_json = list(session_folder.glob("*platform.json"))[0]
    print(platform_json)

    file_splitting_json = list(session_folder.glob("MESOSCOPE_FILE_*"))[0]
    with open(platform_json, "r") as j:
        platform_data = json.load(j)
    sync_file = [i for i in session_folder.glob(platform_data["sync_file"])][0]
    print(sync_file)

    # try to get the framerate from the platform file else use sync file
    try:
        frame_rate_hz = platform_data["imaging_plane_groups"][0]["acquisition_framerate_Hz"]
    except KeyError:
        frame_rate_hz = get_frame_rate_from_sync(str(sync_file), platform_data)

    frame_rate = frame_rate_hz
    
    print("frame rate:" + str(frame_rate))

    with h5py.File(h5_file, "r") as h5_pointer:
        data_pointer = h5_pointer[dataset_name]
        print(data_pointer.shape)
        # in principle, this is done once to avoid loading the data multiple times
        # We generate a few smaller version of the big movie to use for metrics
        cropped_video = subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=1,
            crop=args.crop,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    
    scan = cropped_video
    cropped_video = cropped_video[:, ::2, ::2] + cropped_video[:, 1::2, 1::2] + cropped_video[:, 1::2, ::2] + cropped_video[:, ::2, 1::2]
    print(cropped_video.shape)

    qs = compute_sensitivity(cropped_video.transpose(1, 2, 0), count_weight_gamma=0.1)

    print(qs)

    fig = plt.figure(figsize=(6, 2))
    gs = fig.add_gridspec(
        1, 1,
        left=0.1, right=0.20, bottom=0.55, top=0.85)

    ax = fig.add_subplot(gs[0])


    # Panel A
    m = cropped_video.mean(axis=0)
    _ = ax.imshow(m, vmin=0, vmax=np.quantile(m, 0.999), cmap='gray')
    ax.axis(False)
    ax.set_title('mean fluorescence')
    ax.title.set_size(5)

    # Panel B
    from matplotlib.ticker import FormatStrFormatter
    matplotlib.rc('font', family='sans', size=5)

    x = np.arange(qs["min_intensity"], qs["max_intensity"])

    gs = fig.add_gridspec(
        2, 1, height_ratios=(5, 1),
        left=0.05, right=0.2, bottom=0.15, top=0.45, hspace = 0.05)

    ah = fig.add_subplot(gs[1])
    ah.yaxis.tick_right()
    ah.plot(x/binning, qs['counts'], 'k')
    ah.spines['top'].set_visible(False)
    ah.spines['right'].set_visible(False)
    #ah.spines['bottom'].set_visible(False)
    ah.spines['left'].set_visible(False)
    ah.set_ylabel('density')
    ah.set_xlabel('intensity')
    ah.set_yticks([0])
    ah.grid(True)

    ax = fig.add_subplot(gs[0])
    ax.yaxis.tick_right()
    fit = qs["model"].predict(x.reshape(-1, 1))
    ax.scatter(x/binning, np.float64(np.minimum(fit[-1]*2, qs["variance"])/binning), s=1, color='k', alpha=0.5)
    ax.plot(x / binning, fit / binning, 'red', lw=1, alpha=1)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
    ax.spines['bottom'].set_visible(True)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax.grid(True)
    ax.set_ylabel('variance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('sensitivity={sensitivity:0.1f}; zero level={zero_level:0.0f}'.format(**qs))
    ax.title.set_size(5)

    # Panel C
    gs = fig.add_gridspec(
        1, 1, 
        left=0.30, right=0.45, bottom=0.55, top=0.85)
    ax = fig.add_subplot(gs[0])
    v = ((cropped_video[1:,:,:].astype('float64') - cropped_video[:-1,:,:]) ** 2/2).mean(axis=0)
    imx = np.stack(((m-qs['zero_level'])/qs['sensitivity'], v/qs['sensitivity']/qs['sensitivity'], (m-qs['zero_level'])/qs['sensitivity']), axis=-1)
    img = ax.imshow(np.minimum(1, np.sqrt(0.01 + np.maximum(0, imx/np.quantile(imx, 0.9999))) - 0.1), cmap='PiYG')

    cax = fig.add_axes([0.47, 0.6, 0.025, 0.3])
    cbar = plt.colorbar(img, cax=cax, ticks=[0.2, .5, 0.8], shrink = 0.5)
    cbar.ax.set_yticklabels(['<< 1', '1', '>> 1'])  
    ax.axis(False)
    ax.set_title('coefficient of variation')
    ax.title.set_size(5)

    # Panel D
    # segmentation and trace extraction
    flux = (scan - qs['zero_level']) / qs['sensitivity']

    gs = fig.add_gridspec(
        1, 1, 
        left=0.3, right=0.45, bottom=0.15, top=0.45)
    ax = fig.add_subplot(gs[0])

    im = flux.max(axis=0)
    mx = np.quantile(im, 0.999)
    im[im<0] = np.nan
    img = ax.imshow(im, vmin=-0.0*mx, vmax=mx, cmap=cc.cm.CET_R4)

    cax = fig.add_axes([0.47, 0.1, 0.025, 0.3])

    plt.colorbar(img, cax=cax, shrink=0.5)
    ax.axis(False)
    ax.set_title('max flux (pixel$^{-1}$frame$^{-1}$)');
    ax.title.set_size(5)

    # Panel E
    # make compression lookup tables
    zero = np.int16(np.round(qs['zero_level']))
    LUT1, LUT2 = make_luts(
        zero_level=0, 
        sensitivity=qs['sensitivity'],
        input_max=scan.max() - zero,
        beta=0.5
    )

    gs = fig.add_gridspec(
        2, 2, 
        left=0.60, right=0.9, bottom=0.05, top=0.85, height_ratios = [1.5, 1], hspace=0.5, wspace=0.2)

    ax = fig.add_subplot(gs[0])
    ax.plot(LUT1)
    ax.grid(True)
    ax.set_title('comp. LUT')

    ax = fig.add_subplot(gs[1])
    ax.plot(LUT2[LUT1])
    ax.plot(np.r_[:LUT1.size], np.r_[:LUT1.size], 'k:')
    ax.grid(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.set_title('comp/decomp. transform')

    ax = fig.add_subplot(gs[2])
    frame = np.maximum(0, np.minimum(scan[300,:,:], LUT1.size-1))
    ax.imshow(frame, cmap=cc.cm.CET_R4)
    ax.axis(False)
    ax.set_title('original frame')

    ax = fig.add_subplot(gs[3])
    ax.imshow(LUT2[LUT1[frame]], cmap=cc.cm.CET_R4)
    ax.axis(False)
    ax.set_title('compressed-decompressed')
    fig.savefig(os.path.join(output_dir, 'all_panels.png'), dpi=300)

    compressed = lookup(scan - zero, LUT1)

    gif_path = os.path.join(output_dir, 'compressed.gif')
    save_movie(compressed, gif_path, scale=255//np.max(compressed))        
    print(f'Compression ratio: {np.prod(scan.shape)*2 / os.path.getsize(gif_path):0.2f}')

    qs['compression_ratio'] = np.prod(scan.shape)*2 / os.path.getsize(gif_path)

    # We remove non serializable objects. 
    metrics = {}
    metrics["max_intensity"] = int(qs["max_intensity"])
    metrics["min_intensity"] = int(qs["min_intensity"])
    metrics["sensitivity"] = qs["sensitivity"]
    metrics["zero_level"] = qs["zero_level"]
    metrics["experiment_id"] = experiment_id
    metrics["frame_rate"] = frame_rate

    # We save the metrics to a json file
    with open(os.path.join(output_dir, "photoncompression_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
