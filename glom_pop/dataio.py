"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import functools
import inspect
import os
import shutil
import glob
import warnings

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import yaml
import xml.etree.ElementTree as ET
import pims
from sewar.full_ref import rmse as sewar_rmse
from scipy.signal import resample, savgol_filter
from scipy.interpolate import interp1d
from skimage import filters

from glom_pop import util
from visanalysis.util import h5io
from visanalysis.analysis.imaging_data import ImagingDataObject

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Config settings  # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_config_file():
    path_to_config_file = os.path.join(inspect.getfile(util).split('glom_pop')[0], 'glom_pop', 'config.yaml')
    with open(path_to_config_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def get_included_gloms():
    return get_config_file()['included_gloms']


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Image processing # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def merge_channels(ch1, ch2):
    """
    Merge two channel brains into single array.

    ch1, ch2: np array single channel brain (dims)

    return
        merged np array, 2 channel brain (dims, c)

    """
    return np.stack([ch1, ch2], axis=-1)  # c is last dimension


def save_transforms(registration_object, transform_dir):
    """Save transforms from ANTsPy registration."""
    os.makedirs(os.path.join(transform_dir, 'forward'), exist_ok=True)
    os.makedirs(os.path.join(transform_dir, 'inverse'), exist_ok=True)

    shutil.copy(registration_object['fwdtransforms'][0], os.path.join(transform_dir, 'forward', 'warp.nii.gz'))
    shutil.copy(registration_object['fwdtransforms'][1], os.path.join(transform_dir, 'forward', 'affine.mat'))

    shutil.copy(registration_object['invtransforms'][1], os.path.join(transform_dir, 'inverse', 'warp.nii.gz'))
    shutil.copy(registration_object['invtransforms'][0], os.path.join(transform_dir, 'inverse', 'affine.mat'))


def get_transform_list(transform_dir, direction='forward'):
    """Get transform list from directory, based on direction of transform."""
    if direction == 'forward':
        transform_list = [os.path.join(transform_dir, 'forward', 'warp.nii.gz'),
                          os.path.join(transform_dir, 'forward', 'affine.mat')]
    elif direction == 'inverse':
        transform_list = [os.path.join(transform_dir, 'inverse', 'affine.mat'),
                          os.path.join(transform_dir, 'inverse', 'warp.nii.gz')]

    return transform_list

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Bruker / Prairie View metadata functions # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_bruker_metadata(file_path):
    """
    Parse Bruker / PrairieView metadata from .xml file.

    file_path: .xml filepath
    returns
        metadata: dict
    """
    root = ET.parse(file_path).getroot()

    metadata = {}
    for child in list(root.find('PVStateShard')):
        if child.get('value') is None:
            for subchild in list(child):
                new_key = child.get('key') + '_' + subchild.get('index')
                new_value = subchild.get('value')
                metadata[new_key] = new_value

        else:
            new_key = child.get('key')
            new_value = child.get('value')
            metadata[new_key] = new_value

    metadata['version'] = root.get('version')
    metadata['date'] = root.get('date')
    metadata['notes'] = root.get('notes')

    # Get axis dims
    sequences = root.findall('Sequence')
    c_dim = len(sequences[0].findall('Frame')[0].findall('File'))  # number of channels
    x_dim = metadata['pixelsPerLine']
    y_dim = metadata['linesPerFrame']

    if root.find('Sequence').get('type') == 'TSeries Timed Element':  # Plane time series
        t_dim = len(sequences[0].findall('Frame'))
        z_dim = 1
    elif root.find('Sequence').get('type') == 'TSeries ZSeries Element':  # Volume time series
        t_dim = len(sequences)
        z_dim = len(sequences[0].findall('Frame'))
    elif root.find('Sequence').get('type') == 'ZSeries':  # Single Z stack (anatomical)
        t_dim = 1
        z_dim = len(sequences[0].findall('Frame'))
    else:
        print('!Unrecognized series type in PV metadata!')

    metadata['image_dims'] = [int(x_dim), int(y_dim), z_dim, t_dim, c_dim]

    # get frame times
    if root.find('Sequence').get('type') == 'TSeries Timed Element':  # Plane time series
        frame_times = [float(fr.get('relativeTime')) for fr in root.find('Sequence').findall('Frame')]
        metadata['frame_times'] = frame_times
        metadata['sample_period'] = np.mean(np.diff(frame_times))

    elif root.find('Sequence').get('type') == 'TSeries ZSeries Element':  # Volume time series
        middle_frame = int(len(root.find('Sequence').findall('Frame')) / 2)
        frame_times = [float(seq.findall('Frame')[middle_frame].get('relativeTime')) for seq in root.findall('Sequence')]
        metadata['frame_times'] = frame_times
        metadata['sample_period'] = np.mean(np.diff(frame_times))

    return metadata

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Interacting with hdf5 data file  # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def attach_responses(file_path, series_number, mask, meanbrain, responses, mask_vals,
                     response_set_name='glom', voxel_responses=None):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(h5io.find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        parent_roi_group = epoch_run_group.require_group('aligned_response')
        current_roi_group = parent_roi_group.require_group(response_set_name)

        overwrite_dataset(current_roi_group, 'mask', mask)
        overwrite_dataset(current_roi_group, 'response', responses)
        overwrite_dataset(current_roi_group, 'meanbrain', meanbrain)

        if voxel_responses is not None:
            for ind, vr in enumerate(voxel_responses):
                overwrite_dataset(current_roi_group, 'voxel_resp_{}'.format(mask_vals[ind]), vr)

        current_roi_group.attrs['mask_vals'] = mask_vals


def overwrite_dataset(group, name, data):
    if group.get(name):
        del group[name]
    group.create_dataset(name, data=data)


def get_ft_datapath(ID, ft_dir):
    series_number = ID.series_number
    file_name = os.path.split(ID.file_path)[-1].split('.')[0]
    look_path = os.path.join(ft_dir,
                             file_name.replace('-', ''),
                             'series{}'.format(str(series_number).zfill(3)))
    glob_res = glob.glob(os.path.join(look_path, '*.dat'))
    if len(glob_res) == 0:
        return False
    elif len(glob_res) > 1:
        print('Warning! Multiple FT .dat files found at {}'.format(look_path))
        print('Returning first file in list: {}'.format(sorted(glob_res)[0]))
        return sorted(glob_res)[0]
    elif len(glob_res) == 1:
        return glob_res[0]



def process_fictrac_data(ft_data, 
                         exclude_thresh=None, 
                         timestamps=None, 
                         ball_diameter=9, 
                         filter_duration=0.75, 
                         filter_polyorder=3):
    """
    Process raw Fictrac data: handle timestamps, calculate velocities, filter.
    
    Args:
        ft_data: DataFrame with Fictrac data
        exclude_thresh: threshold for exclusion
        timestamps: optional aligned timestamps to use instead of Fictrac's [sec]
            If provided, will be used as-is and not modified.
            If not provided, will use Fictrac timestamps, relative to first timestamp.
        ball_diameter: diameter of ball in mm
        filter_duration: duration of the Savitzky-Golay filter window in seconds (default: 0.75)
        filter_polyorder: order of the Savitzky-Golay filter (default: 3)
    Returns:
        dict with processed data
    """
    # 1. Handle Timestamps
    if len(timestamps) != len(ft_data):
        warnings.warn('DataIO: Timestamps do not match Fictrac data length. Using Fictrac timestamps.')
        timestamps = None
    
    if timestamps is not None:
        print('DataIO: Using provided timestamps instead of Fictrac timestamps.')
    else:
        # Try to find timestamp column
        if 'timestamp' in ft_data.columns:
            ts = ft_data['timestamp'].values
        elif ft_data.shape[1] > 21:
            ts = ft_data.iloc[:, 21].values
        else:
            # Fallback or error?
            ts = np.arange(len(ft_data)) # Dummy
        
        # Safe convert
        if ts.dtype == object:
            ts = pd.to_numeric(ts, errors='coerce')
            
        # Robust Timestamp Logic
        is_unix = ts > 1e10
        
        if np.all(is_unix):
            # Case 1: All Unix
            pass
        elif is_unix[0] and not np.any(is_unix[1:]):
            # Case 2: First only is Unix
            # make ts all epoch time
            ts[1:] += ts[0]
        else:
            # Case 3: Relative
            pass
            
        timestamps = ts / 1e3 # ms -> sec

        # Use relative timestamps with respect to first timestamp
        if len(timestamps) > 0:
            timestamps = timestamps - timestamps[0]

    # 2. Calculate Velocities
    # Try named columns first
    if 'rel_vec_world_y' in ft_data.columns and 'rel_vec_world_z' in ft_data.columns:
        y_rot = ft_data['rel_vec_world_y'].values
        z_rot = ft_data['rel_vec_world_z'].values
        x_rot = ft_data['rel_vec_world_x'].values if 'rel_vec_world_x' in ft_data.columns else np.zeros_like(y_rot)
    else:
        # Fallback to indices (standard Fictrac)
        # 6, 7, 8 are rot angles? No, 6,7,8 are usually x,y,z?
        # dataio.py used 5, 6, 7 for x, y, z
        x_rot = ft_data.iloc[:, 5].values
        y_rot = ft_data.iloc[:, 6].values
        z_rot = ft_data.iloc[:, 7].values

    # Calculate FPS for velocity conversion
    fps = 1 / np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 50.0

    xrot_deg = np.rad2deg(x_rot) * fps
    yrot_deg = np.rad2deg(y_rot) * fps
    zrot_deg = np.rad2deg(z_rot) * fps

    # Calculate window length based on duration
    window_length = int(filter_duration * fps)
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure window_length is at least polyorder + 2
    if window_length < filter_polyorder + 2:
        print(f'Warning: Filter window length ({window_length}) is too short for polyorder ({filter_polyorder}). Setting to polyorder + 2.')
        window_length = filter_polyorder + 2
    
    xrot_filt = savgol_filter(xrot_deg, window_length, filter_polyorder)
    yrot_filt = savgol_filter(yrot_deg, window_length, filter_polyorder)
    zrot_filt = savgol_filter(zrot_deg, window_length, filter_polyorder)

    walking_mag = np.sqrt(xrot_filt**2 + yrot_filt**2 + zrot_filt**2)

    if exclude_thresh is not None:
        if isinstance(exclude_thresh, (int, float)):
            exclude_mask = walking_mag > exclude_thresh
        elif isinstance(exclude_thresh, (list, np.ndarray)) and len(exclude_thresh) == 3:
            exclude_mask = (np.abs(xrot_filt) > exclude_thresh[0]) | \
                                    (np.abs(yrot_filt) > exclude_thresh[1]) | \
                                    (np.abs(zrot_filt) > exclude_thresh[2])
        elif callable(exclude_thresh):
            exclude_mask = exclude_thresh(x=xrot_filt, y=yrot_filt, z=zrot_filt)
        else:
            raise ValueError('Unrecognized exclude_thresh: {}'.format(exclude_thresh))
        
        print(f'process_fictrac_data: Excluding {np.sum(exclude_mask) / len(exclude_mask) * 100:.2f}% of timepoints based on exclude_thresh.')
        xrot_filt[exclude_mask] = np.nan
        yrot_filt[exclude_mask] = np.nan
        zrot_filt[exclude_mask] = np.nan
        walking_mag[exclude_mask] = np.nan

    ball_circumference = np.pi * ball_diameter  # mm
    fwd_vel = (yrot_filt/360) * ball_circumference  # deg/sec --> mm/sec
    lat_vel = (xrot_filt/360) * ball_circumference  # deg/sec --> mm/sec (leftward is positive)
    turning_vel = zrot_filt  # deg/sec (left turn / clockwise is positive)
    
    return {
        'timestamps': timestamps,
        'fwd_vel': fwd_vel, # mm/sec
        'lat_vel': lat_vel, # mm/sec (leftward is positive)
        'turning_vel': turning_vel, # deg/sec (left turn / clockwise is positive)
        'walking_mag': walking_mag, # deg/sec
        'xrot_filt': xrot_filt, # deg/sec
        'yrot_filt': yrot_filt, # deg/sec
        'zrot_filt': zrot_filt # deg/sec
    }

def load_fictrac_data(ID:ImagingDataObject, 
                      ft_data_path, 
                      timestamps=None,
                      exclude_thresh=None, 
                      binarizing_var_name='walking_mag',
                      normalization='none',
                      baseline_period='pre',
                      ball_diameter=9,
                      show_qc=True):
    """
    Load and process FicTrac data from .dat file.
    Args:
        ID: ImagingDataObject
        ft_data_path: path to FicTrac .dat file
        timestamps: timestamps to use instead of the ones in the FicTrac file
        exclude_thresh: threshold for excluding high rotation values (deg/sec)
            If None, no exclusion is performed.
            Any time point with walking_mag above this threshold in magnitude is set to nan.
        binarizing_var_name: variable to use for thresholding and binarizing behavior
            Options: 'walking_mag', 'fwd_vel', 'turning_vel'
        normalization: method to normalize the signal
            'dff': convert from raw intensity value to dF/F based on mean of pre_time
            'zscore': z-score the signal based on mean and std of pre_time
            'mean_subtraction': subtract the mean of baseline period from the signal
            'none': no normalization, use raw signal
        baseline_period: (str or list) Period to use for calculating baseline
            'pre': use pre_time period only for baseline
            'whole': use entire epoch (pre + stim + post) for baseline
            list of 2-tuples: e.g. [(-2.0, -0.5), (5.0, 6.0)] specifying time
                                windows in seconds relative to stimulus onset (0s).
                                Pre-stimulus times are negative.
        ball_diameter: diameter of ball in mm
        show_qc: whether to show QC plots
    Returns:
        behavior_data: dict with behavior data
    """
    # Imaging time stamps
    imaging_time_vector = ID.getResponseTiming()['time_vector']

    # exclude_thresh: deg per sec
    ft_data = pd.read_csv(ft_data_path, header=None)
    
    # Use shared processing logic
    processed = process_fictrac_data(ft_data, exclude_thresh=exclude_thresh, timestamps=timestamps, ball_diameter=ball_diameter)
    
    timestamps = processed['timestamps']
    fwd_vel = processed['fwd_vel']
    turning_vel = processed['turning_vel']
    walking_mag = processed['walking_mag']
    xrot_filt = processed['xrot_filt']
    yrot_filt = processed['yrot_filt']
    zrot_filt = processed['zrot_filt']

    # Downsample from camera frame rate to imaging frame rate
    # fwd_vel_ds = resample(fwd_vel, len(imaging_time_vector))
    # turning_vel_ds = resample(turning_vel, len(imaging_time_vector))
    # walking_mag_ds = resample(walking_mag, len(imaging_time_vector))
    
    def downsample_to_imaging_rate(variable, timestamps, imaging_time_vector):
        # # Convert to pandas Series with datetime index
        # df_raw = pd.Series(variable, index=pd.to_datetime(timestamps))
        # target_index = pd.to_datetime(imaging_time_vector)
        # print(target_index)
        # freq_str = pd.infer_freq(target_index)
        # if freq_str is None:
        #     print('Warning: Could not infer frequency of imaging time vector. Using mean sampling interval instead.')
        # # 1. Resample and Aggregate (Mean ignores NaNs by default)
        # # 'label' and 'closed' determine if the bin starts or ends at the timestamps
        # resampled = df_raw.resample(freq_str, label='left', closed='left').mean()
        # # 2. Reindex to force strict alignment with your numpy array
        # # This handles cases where resample might skip empty periods or add extra ones
        # final_result = resampled.reindex(target_index)
        # # 3. Extract the values back to numpy
        # downsampled_values = final_result.to_numpy()

        # Approach: Interpolate to fill NaNs, then resample
        y_filled = pd.Series(variable).interpolate(method='linear').to_numpy()
        downsampled_values = resample(y_filled, len(imaging_time_vector))
        return downsampled_values

    fwd_vel_ds = downsample_to_imaging_rate(fwd_vel, timestamps, imaging_time_vector)
    turning_vel_ds = downsample_to_imaging_rate(turning_vel, timestamps, imaging_time_vector)
    walking_mag_ds = downsample_to_imaging_rate(walking_mag, timestamps, imaging_time_vector)

    print(f"Pre-ds Nan values: {np.sum(np.isnan(turning_vel))} / {np.size(turning_vel)}")
    print(f"Post-ds Nan values: {np.sum(np.isnan(turning_vel_ds))} / {np.size(turning_vel_ds)}")

    if binarizing_var_name == 'walking_mag':
        binarizing_var = walking_mag
        binarizing_var_ds = walking_mag_ds

        thresh = filters.threshold_li(binarizing_var)
        binary_behavior = (binarizing_var > thresh).astype('int')
        binary_behavior_ds = (binarizing_var_ds > thresh).astype('int')
    elif binarizing_var_name == 'fwd_vel':
        binarizing_var = fwd_vel
        binarizing_var_ds = fwd_vel_ds

        thresh = filters.threshold_li(binarizing_var)
        binary_behavior = (binarizing_var > thresh).astype('int')
        binary_behavior_ds = (binarizing_var_ds > thresh).astype('int')
    elif binarizing_var_name == 'turning_vel':
        binarizing_var = turning_vel
        binarizing_var_ds = turning_vel_ds

        thresh = filters.threshold_li(np.abs(binarizing_var))
        binary_behavior = (np.abs(binarizing_var) > thresh).astype('int')
        binary_behavior_ds = (np.abs(binarizing_var_ds) > thresh).astype('int')
    else:
        raise ValueError('Unrecognized binarizing_var_name: {}'.format(binarizing_var_name))

    _, behavior_binary_matrix = ID.getEpochResponseMatrix(binary_behavior_ds[np.newaxis, :],
                                                          normalization=normalization, baseline_period=baseline_period)

    _, walking_response_matrix = ID.getEpochResponseMatrix(walking_mag_ds[np.newaxis, :],
                                                           normalization=normalization, baseline_period=baseline_period)

    _, turning_vel_response_matrix = ID.getEpochResponseMatrix(turning_vel_ds[np.newaxis, :],
                                                           normalization=normalization, baseline_period=baseline_period)

    _, fwd_vel_response_matrix = ID.getEpochResponseMatrix(fwd_vel_ds[np.newaxis, :],
                                                         normalization=normalization, baseline_period=baseline_period)

    is_behaving = ID.getResponseAmplitude(behavior_binary_matrix, metric='mean') > 0.25
    walking_amp = ID.getResponseAmplitude(walking_response_matrix, metric='mean')
    turning_vel_amp = ID.getResponseAmplitude(turning_vel_response_matrix, metric='mean')
    fwd_vel_amp = ID.getResponseAmplitude(fwd_vel_response_matrix, metric='mean')

    # Peak walking during trial
    smth = savgol_filter(walking_response_matrix, 5, 3, axis=-1)
    walking_peak = ID.getResponseAmplitude(smth, metric='max')

    behavior_data = {'walking_mag': walking_mag,  # n video frames
                     'walking_mag_ds': walking_mag_ds,  # n imaging frames
                     'fwd_vel': fwd_vel,  # mm/sec, shape=n video frames
                     'fwd_vel_ds': fwd_vel_ds,  # mm/sec, shape=n imaging frames
                     'turning_vel': turning_vel,  # deg/sec, shape=n video frames
                     'turning_vel_ds': turning_vel_ds,  # deg/sec, shape=n imaging frames
                     'behavior_binary_matrix': behavior_binary_matrix,  # 1 x trials x time
                     'walking_response_matrix': walking_response_matrix,  # 1 x trials x time
                     'turning_vel_response_matrix': turning_vel_response_matrix,  # 1 x trials x time
                     'fwd_vel_response_matrix': fwd_vel_response_matrix,  # 1 x trials x time
                     'walking_amp': walking_amp,  # n trials
                     'walking_peak': walking_peak,
                     'turning_vel_amp': turning_vel_amp,
                     'fwd_vel_amp': fwd_vel_amp,
                     'is_behaving': is_behaving,  # n trials
                     'thresh': thresh,
                     'timestamps': timestamps,  # sec
                     'xrot_filt': xrot_filt,
                     'yrot_filt': yrot_filt,
                     'zrot_filt': zrot_filt,
                     }

    if show_qc:
        fh, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].plot(binarizing_var)
        ax[0].axhline(thresh, color='r')
        ax[1].hist(binarizing_var, 100)
        ax[1].axvline(thresh, color='r')
        ax[0].set_title(f'{os.path.split(ID.file_path)[-1]}: {ID.series_number}')

        return behavior_data, fh
    else:
        return behavior_data


def load_responses(ID:ImagingDataObject, response_set_name='glom', get_erm=True, get_voxel_responses=False, normalization='dff', baseline_period='pre'):
    """
    Load responses from h5 file

    Args:
        ID: ImagingDataObject
        response_set_name: name of response set in the h5 file
        get_erm: whether to get the epoch response matrix
        get_voxel_responses: whether to get voxel responses
        normalization: method to normalize the signal
            'dff': convert from raw intensity value to dF/F based on mean of pre_time (default)
            'zscore': z-score the signal based on mean and std of pre_time
            'mean_subtraction': subtract the mean of baseline period from the signal
            'none': no normalization, use raw signal
        baseline_period: (str or list) Period to use for calculating baseline
            'pre': use pre_time period only for baseline (default)
            'whole': use entire epoch (pre + stim + post) for baseline
            list of 2-tuples: e.g. [(-2.0, -0.5), (5.0, 6.0)] specifying time
                                windows in seconds relative to stimulus onset (0s).
                                Pre-stimulus times are negative.

    Returns:
        response_data: dict with response data
    """
    response_data = {}
    with h5py.File(ID.file_path, 'r') as experiment_file:
        find_partial = functools.partial(h5io.find_series, sn=ID.series_number)
        roi_parent_group = experiment_file.visititems(find_partial)['aligned_response']
        roi_set_group = roi_parent_group[response_set_name]
        response_data['response'] = roi_set_group.get("response")[:]
        response_data['mask'] = roi_set_group.get("mask")[:]
        response_data['meanbrain'] = roi_set_group.get("meanbrain")[:]
        response_data['mask_vals'] = roi_set_group.attrs['mask_vals']

        if get_voxel_responses:
            mask_vals = np.unique(response_data['mask'])[1:].astype(int)  # exclude first (0)
            voxel_responses = {}
            voxel_epoch_responses = {}
            for mv in mask_vals:
                voxel_responses[mv] = roi_set_group.get('voxel_resp_{}'.format(mv))[:].astype('float32')
                _, response_matrix = ID.getEpochResponseMatrix(voxel_responses[mv], normalization=normalization, baseline_period=baseline_period)
                voxel_epoch_responses[mv] = response_matrix

    if get_voxel_responses:
        response_data['voxel_responses'] = voxel_responses
        response_data['voxel_epoch_responses'] = voxel_epoch_responses

    if get_erm:
        # epoch_response matrix for glom responses
        time_vector, response_matrix = ID.getEpochResponseMatrix(response_data.get('response'), normalization=normalization, baseline_period=baseline_period)
        response_data['epoch_response'] = response_matrix  # shape = (gloms, trials, time)
        response_data['time_vector'] = time_vector

    return response_data


def get_glom_mask_decoder(mask):
    sync_dir = get_config_file()['sync_dir']
    # Load mask key for VPN types
    vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))

    vals = np.unique(mask)[1:]  # exclude first val (=0, not a glom)

    names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(vals), 'vpn_types']
    return vals, names


def get_glom_name_from_val(val):
    sync_dir = get_config_file()['sync_dir']
    # Load mask key for VPN types
    vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))
    name = vpn_types.iloc[np.where(vpn_types['Unnamed: 0'] == val)[0], 1].values[0]

    return name


def get_glom_vals_from_names(glom_names):
    sync_dir = get_config_file()['sync_dir']
    vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))
    vals = np.array([vpn_types.iloc[np.where(vpn_types.vpn_types == ig)[0][0], 0] for ig in glom_names])

    return vals


def filter_epoch_response_matrix(response_data, included_vals, glom_size_threshold=10):
    # epoch_response_matrix: shape=(gloms, trials, time)
    epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
    epoch_response_matrix[:] = np.nan

    for val_ind, included_val in enumerate(included_vals):
        new_glom_size = np.sum(response_data.get('mask') == included_val)

        if new_glom_size > glom_size_threshold:
            mask_vals = [int(mv.split('_')[0]) for mv in response_data.get('mask_vals')]
            pull_ind = np.where(included_val == mask_vals)[0][0]
            epoch_response_matrix[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
        else:  # Exclude because this glom, in this fly, is too tiny
            pass

    return epoch_response_matrix


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Interacting with behavior video  # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_frame_size(filepath):
    frame = pims.as_grey(pims.Video(filepath))[0]
    return frame.shape


def get_video_timing(frame_triggers, sample_rate=10000):
    # Use frame trigger voltage output to find frame times in bruker time
    # shift & normalize so trace lives on [0 1]
    frame_triggers = frame_triggers - np.min(frame_triggers)
    frame_triggers = frame_triggers / np.max(frame_triggers)

    # find trigger up times
    threshold = 0.5
    V_orig = frame_triggers[0:-2]
    V_shift = frame_triggers[1:-1]
    frame_times = np.where(np.logical_and(V_orig < threshold, V_shift >= threshold))[0] + 1

    frame_times = frame_times / sample_rate  # Seconds

    print('{} frame triggers sent'.format(frame_times.shape[0]))

    return frame_times


def get_ball_movement(filepath,
                      cropping=((90, 0), (10, 20), (0, 0)),  # Pixels to trim from ((L, R), (T, B), (RGB_start, RGB_end))
                      ):
    # Load and crop vid as a pims object
    whole_vid = pims.as_grey(pims.Video(filepath))
    cropped_vid = pims.as_grey(pims.process.crop(pims.Video(filepath), cropping))

    # Measure ball movement by computing rmse between successive frames
    ball_rmse = np.array([sewar_rmse(cropped_vid[f], cropped_vid[f+1]) for f in range(len(cropped_vid)-1)])

    # Binarize using Otsu threshold
    # i.e. minimize within-class variance
    thresh = filters.threshold_otsu(ball_rmse)
    binary_behavior = ball_rmse > thresh

    print('{} frames in movie'.format(ball_rmse.shape[0]+1))

    video_results = {'frame': whole_vid[1],
                     'cropped_frame': cropped_vid[1],
                     'rmse': ball_rmse,
                     'binary_behavior': binary_behavior,
                     'binary_thresh': thresh
                     }

    return video_results


def attach_behavior_data(file_path,
                         series_number,
                         video_results):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(h5io.find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        behavior_group = epoch_run_group.require_group('behavior')

        for key in video_results:
            overwrite_dataset(behavior_group, key, video_results[key])
