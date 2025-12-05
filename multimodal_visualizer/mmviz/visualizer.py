import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import h5py
import napari
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QDockWidget
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
import functools
from scipy.signal import savgol_filter
from .widgets import TimeSeriesPlotter, VideoViewer, DataLoader, PlaybackControl, StimulusViewer

from visanalysis.analysis.imaging_data import ImagingDataObject
from visanalysis.util import h5io
from . import utils

class MultiModalVisualizer:
    def __init__(self, no_brain=False, series_name=None):
        self.viewer = napari.Viewer(title="Multi-modal Data Visualizer")
        self.no_brain = no_brain
        self.series_name = series_name
        
        # Data containers
        self.brain_data = None
        self.brain_metadata = None
        self.fictrac_data = None
        self.stimulus_data = None
        self.video_data = None
        self.video_timestamps_unix = None
        self.video_timestamps_relative = None
        self.start_time_unix = None # Use camera start time for now...
        
        # Setup GUI
        self.setup_widgets()
        
        # Connect viewer events
        self.viewer.dims.events.current_step.connect(self.on_time_change)

    def setup_widgets(self):
        # Data Loader Widget
        self.loader_widget = DataLoader(self)
        self.viewer.window.add_dock_widget(self.loader_widget, area='right', name='Data Loader')
        
        # Video Viewer Widget
        self.video_widget = VideoViewer()
        self.viewer.window.add_dock_widget(self.video_widget, area='right', name='Behavior Video')
        
        # Time Series Plotter Widget
        self.plotter_widget = TimeSeriesPlotter()
        self.plotter_widget.timeChanged.connect(self.on_plot_time_change)
        self.viewer.window.add_dock_widget(self.plotter_widget, area='bottom', name='Time Series')

        # Playback Control (Top Right)
        self.playback_widget = PlaybackControl()
        self.playback_widget.btn_play.clicked.connect(self.play_video)
        self.playback_widget.btn_pause.clicked.connect(self.pause_video)
        self.playback_widget.btn_stop.clicked.connect(self.stop_video)
        self.playback_widget.slider_speed.valueChanged.connect(self.set_playback_speed)
        self.viewer.window.add_dock_widget(self.playback_widget, area='right', name='Playback')

        # Stimulus Viewer (Bottom)
        self.stimulus_widget = StimulusViewer()
        self.plotter_widget.sigEpochChanged.connect(self.stimulus_widget.update_epoch)
        self.viewer.window.add_dock_widget(self.stimulus_widget, area='bottom', name='Stimulus Info')
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playback)
        self.playback_speed = 1.0 # Multiplier
        self.current_playback_time = 0.0
        self.fps = 50.0 # Default, will update from data

    def load_data(self, hdf5_path, series_name, video_path=None, fictrac_path=None, brain_path=None, brain_xml_path=None, cam_ts_path=None):
        print(f"Loading data...")
        self.series_name = series_name
        
        # 0. Initialize ImagingDataObject
        self.ID = None
        if hdf5_path and os.path.exists(hdf5_path):
            try:
                # Parse series number
                if isinstance(series_name, int):
                    series_number = series_name
                elif str(series_name).isdigit():
                    series_number = int(series_name)
                else:
                    series_number = int(series_name.split('_')[-1])
                
                self.ID = ImagingDataObject(hdf5_path, series_number, quiet=True)
                print(f"Initialized ImagingDataObject for series {series_number}")
            except Exception as e:
                print(f"Failed to initialize ImagingDataObject: {e}")

        # 1. Load Brain Volume
        if not self.no_brain and brain_path and os.path.exists(brain_path):
            print(f"Loading brain volume from {brain_path}...")
            import nibabel as nib
            img = nib.load(brain_path)
            self.brain_data = img.get_fdata()
            if self.brain_data.ndim == 4:
                self.brain_data = self.brain_data.transpose(3, 2, 1, 0) # t, z, y, x
            
            self.viewer.add_image(self.brain_data, name='Brain Volume', colormap='gray_translucent')
        else:
            print("Skipping brain volume loading.")
        
        # 2. Load Metadata (Timestamps)
        self.brain_timestamps = None
        # Try loading from ID first
        if self.ID:
            try:
                response_timing = self.ID.getResponseTiming()
                if 'time_vector' in response_timing:
                    self.brain_timestamps = response_timing['time_vector']
                    print("Loaded brain timestamps from ImagingDataObject.")
            except Exception as e:
                print(f"Could not load timestamps from ID: {e}")

        # Fallback to XML
        if self.brain_timestamps is None and brain_xml_path and os.path.exists(brain_xml_path):
            print(f"Loading metadata from {brain_xml_path}...")
            self.brain_metadata = utils.get_bruker_metadata(brain_xml_path)
            self.brain_timestamps = np.array(self.brain_metadata['frame_times'])
        elif self.brain_timestamps is None:
            print("No metadata (XML) loaded.")
        
        # 3. Load Video
        if video_path and os.path.exists(video_path):
            print(f"Loading Video from {video_path}...")
            self.video_widget.load_video(video_path)
        else:
            print("No video loaded.")

        # Load camera timestamps
        if cam_ts_path and os.path.exists(cam_ts_path):
            print(f"Loading camera timestamps from {cam_ts_path}...")
            cam_ts_data = np.loadtxt(cam_ts_path) # [frame number, cam timestamp (s), cpu timestamp (unix, s)]
            
            # Use the first cpu timestamp as the start time, and use it to convert camera timestamps to unix timestamps
            self.video_timestamps_relative = cam_ts_data[:, 1] - cam_ts_data[0, 1]
            self.video_timestamps_unix = cam_ts_data[0, 2] + self.video_timestamps_relative
            self.start_time_unix = cam_ts_data[0, 2]
            self.fps = 1 / np.mean(np.diff(self.video_timestamps_relative))
        else:
            print("No camera timestamps loaded.")
            self.video_timestamps_relative = None
            self.video_timestamps_unix = None
            self.start_time_unix = None

        # 4. Load Fictrac
        self.fictrac_data = None
        # Try loading from ID first
        try:
            # Manually read fictrac data from HDF5 using ID context
            # We can't use getBehaviorData easily as it slices by epoch
            # But we can use h5py with ID.file_path
            
            with h5py.File(self.ID.file_path, 'r') as f:
                # Find series group
                find_partial = functools.partial(h5io.find_series, sn=self.ID.series_number)
                series_grp = f.visititems(find_partial)
                
                if series_grp and 'behavior' in series_grp and 'fictrac_data' in series_grp['behavior']:
                    ft_dset = series_grp['behavior']['fictrac_data']
                    header = ft_dset.attrs.get('fictrac_data_header')
                    if header is not None:
                        header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in header]
                        ft_data = pd.DataFrame(ft_dset[:], columns=header)
                        self.fictrac_data = self.process_fictrac_data(ft_data, camera_timestamps=self.video_timestamps_unix)
                        if self.fictrac_data:
                            print("Loaded Fictrac data from ImagingDataObject.")
        except Exception as e:
            print(f"Failed to load Fictrac from ID: {e}")

        if self.fictrac_data is None and fictrac_path and os.path.exists(fictrac_path):
            print(f"Loading Fictrac from {fictrac_path}...")
            self.load_fictrac_direct(fictrac_path, camera_timestamps=self.video_timestamps_unix)
        elif self.fictrac_data is None:
            print("No Fictrac data loaded.")
            self.fictrac_data = {'timestamp': [], 'fwd_vel': [], 'turning_vel': []}

        # 5. Load Stimulus
        print(f"Loading Stimulus from ID...")
        self.load_stimulus_from_id()
        
        # 6. Synchronize and Plot
        self.synchronize_and_plot()
        
        # 7. Update DataLoader UI if it exists
        if hasattr(self, 'loader_widget'):
            file_paths = {
                'video': video_path,
                'fictrac': fictrac_path,
                'brain': brain_path,
                'brain_xml': brain_xml_path,
                'camera_timestamps': cam_ts_path
            }
            self.loader_widget.update_paths(hdf5_path, series_name, file_paths)
        
        print("Data loading complete.")

    def process_fictrac_data(self, ft_data, camera_timestamps=None):
        # 1. Handle Timestamps
        if camera_timestamps is not None and len(camera_timestamps) == len(ft_data):
            print("  Using Camera Timestamps for Fictrac.")
            timestamp = camera_timestamps
        else:
            print("  Using Fictrac Timestamps for Fictrac.")
            if 'timestamp' in ft_data.columns:
                ts = ft_data['timestamp'].values
            else:
                ts = ft_data.iloc[:, 21].values

            # Check for Unix timestamps (large values)

            # Let's do a safe convert if object
            if ts.dtype == object:
                ts = pd.to_numeric(ts, errors='coerce')
                
            # Logic:
            # Case 1: All entries are epoch time (> 1e10)
            # Case 2: Only first entry is epoch time (> 1e10)
            # Case 3: None are epoch time (all relative ms)
            
            is_unix = ts > 1e10
            
            if np.all(is_unix):
                # Case 1: All Unix
                print("  Detected ALL Unix timestamps. Using them directly.")
                
            elif is_unix[0] and not np.any(is_unix[1:]):
                # Case 2: First only is Unix
                print("  Detected Single Unix timestamp at start. Treating the rest as relative ms from start.")
                
                # make ts all epoch time
                ts[1:] += ts[0]

            else: 
                # Bad timestamps
                print("  Bad Fictrac timestamps.")
            
            timestamp = ts / 1e3 # ms -> sec

        # 2. Calculate velocities
        # Try to find columns by name, otherwise by index        
        if 'rel_vec_world_y' in ft_data.columns and 'rel_vec_world_z' in ft_data.columns:
            y_rot = ft_data['rel_vec_world_y'].values
            z_rot = ft_data['rel_vec_world_z'].values
        else:
            y_rot = ft_data.iloc[:, 6].values
            z_rot = ft_data.iloc[:, 7].values

        y_rot_deg = np.rad2deg(y_rot) * self.fps
        z_rot_deg = np.rad2deg(z_rot) * self.fps
        
        # Filter
        window_length = 151
        if len(y_rot_deg) <= window_length:
            window_length = len(y_rot_deg)
            if window_length % 2 == 0: window_length -= 1
        
        if window_length > 3:
            yrot_filt = savgol_filter(y_rot_deg, window_length, 3)
            zrot_filt = savgol_filter(z_rot_deg, window_length, 3)
        else:
            yrot_filt = y_rot_deg
            zrot_filt = z_rot_deg
        
        ball_diameter = 9 # mm
        ball_circumference = np.pi * ball_diameter # mm
        fwd_vel = (yrot_filt/360) * ball_circumference # deg/sec --> mm/sec
        turning_vel = zrot_filt # deg/sec
        
        return {
            'timestamp': timestamp,
            'fwd_vel': fwd_vel,
            'turning_vel': turning_vel
        }

    def load_fictrac_direct(self, filepath, camera_timestamps=None):
        
        # Copied/Adapted from dataio.py
        try:
            ft_data = pd.read_csv(filepath, header=None)
            
            # Ensure numeric for column 21 (timestamp)
            ft_data[21] = pd.to_numeric(ft_data[21], errors='coerce')
            
            # Process
            self.fictrac_data = self.process_fictrac_data(ft_data, camera_timestamps)
            
            if self.fictrac_data is None:
                 print("Error: No valid Fictrac data remaining after filtering.")
                 self.fictrac_data = {'timestamp': [], 'fwd_vel': [], 'turning_vel': []}
                 
        except Exception as e:
            print(f"Error loading Fictrac file: {e}")
            self.fictrac_data = {'timestamp': [], 'fwd_vel': [], 'turning_vel': []}


    def load_stimulus_from_id(self):
        self.stimulus_data = {'epochs': []}
        
        if not self.ID:
            print("No ImagingDataObject initialized.")
            return

        try:
            # Get Epoch Parameters
            epoch_params = self.ID.getEpochParameters()
            
            for ep in epoch_params:
                self.stimulus_data['epochs'].append({
                    'start': ep.get('epoch_unix_time'),
                    'end': ep.get('epoch_end_unix_time'),
                    'name': ep.get('name', 'Unknown'),
                    'parameters': ep
                })
                
            print(f"Loaded {len(self.stimulus_data['epochs'])} epochs via ImagingDataObject.")
            
        except Exception as e:
            print(f"Error loading stimulus with ImagingDataObject: {e}")
            import traceback
            traceback.print_exc()

    def synchronize_and_plot(self):
        # Fictrac
        self.plotter_widget.plot_fictrac(self.fictrac_data['timestamp'], 
                                         self.fictrac_data['fwd_vel'], 
                                         self.fictrac_data['turning_vel'],
                                         t0=self.start_time_unix,
                                         )
                                         
        # Stimulus
        if hasattr(self, 'video_timestamps_unix') and self.video_timestamps_unix is not None:
            # Align stimulus to video start (t=0)
            
            # Stimulus epochs are in Unix time.
            # t_relative = t_unix - start_unix_time
            
            print(f"DEBUG: Aligning stimulus. start_unix_time: {self.start_time_unix}")
            if self.stimulus_data['epochs']:
                print(f"DEBUG: First epoch start: {self.stimulus_data['epochs'][0]['start']}")
            
            aligned_epochs = []
            aligned_epochs = []
            for i, epoch in enumerate(self.stimulus_data['epochs']):
                if epoch['start'] == 0:
                    print(f"DEBUG: Found epoch with start=0: {epoch['name']}")
                
                aligned_start = epoch['start'] - self.start_time_unix
                aligned_end = epoch['end'] - self.start_time_unix
                
                aligned_epochs.append({
                    'index': i,
                    'start': aligned_start,
                    'end': aligned_end,
                    'name': epoch['name'],
                    'parameters': epoch.get('parameters', {}) # Pass parameters
                })
            
            if aligned_epochs:
                starts = [e['start'] for e in aligned_epochs]
                print(f"DEBUG: Aligned epochs range: {min(starts)} to {max(starts)}")
            
            self.plotter_widget.plot_stimulus(aligned_epochs)
        else:
            print("Warning: Could not align stimulus data (missing video unix timestamps).")

    def on_time_change(self, event):
        current_step = event.value[0] # Assuming t is dim 0
        if self.brain_timestamps is not None and current_step < len(self.brain_timestamps):
            current_time = self.brain_timestamps[current_step]
            
            # Update Plotter Line
            self.plotter_widget.update_time_line(current_time)
            
            # Update Video
            # Find closest video frame
            if self.video_timestamps_relative is not None:
                # Find index where timestamp is closest to current_time
                idx = (np.abs(self.video_timestamps_relative - current_time)).argmin()
                self.video_widget.set_frame(idx)

    def on_plot_time_change(self, time):
        # Update Video based on plot time
        if self.video_timestamps_relative is not None:
            # Find index where timestamp is closest to time
            idx = (np.abs(self.video_timestamps_relative - time)).argmin()
            self.video_widget.set_frame(idx)
            
        # Update Napari slider if possible?
        # If we have brain timestamps, we can find the closest step
        if self.brain_timestamps is not None and len(self.brain_timestamps) > 0:
            # Find closest brain frame
            step = (np.abs(self.brain_timestamps - time)).argmin()
            # Avoid infinite loop by checking current step?
            # self.viewer.dims.set_current_step(0, step) # This might trigger on_time_change
            pass

    def play_video(self):
        if not self.timer.isActive():
            # Limit update rate to screen refresh rate to avoid overwhelming the GUI
            # If video FPS is higher (e.g. 200Hz -> 5ms), we don't need to paint every frame.
            
            screen = QApplication.primaryScreen()
            refresh_rate = screen.refreshRate() if screen else 60.0
            if refresh_rate < 1: refresh_rate = 60.0 # Fallback
            
            screen_refresh_interval = int(1000 / refresh_rate)
            
            video_interval = int(1000 / self.fps)
            interval = max(video_interval, screen_refresh_interval)
            
            self.last_update_time = time.time()
            self.timer.start(interval)
            print(f"Playback started. Interval: {interval}ms (Video FPS: {self.fps:.1f}, Screen: {refresh_rate:.1f}Hz)")

    def pause_video(self):
        self.timer.stop()
        print("Playback paused.")

    def stop_video(self):
        self.timer.stop()
        self.current_playback_time = 0.0
        if self.video_timestamps_relative is not None:
             self.current_playback_time = self.video_timestamps_relative[0]
        
        self.on_plot_time_change(self.current_playback_time)
        self.plotter_widget.update_time_line(self.current_playback_time)
        print("Playback stopped.")

    def update_playback(self):
        # Advance time
        # We need to know the current time.
        # Let's assume current_playback_time is tracked or we get it from plot?
        # Better to track it here to avoid loop issues.
        
        # If we just started, sync with plot
        # But wait, on_plot_time_change updates video.
        # We need to get current time from somewhere if user scrubbed while paused.
        # Let's rely on the plot's time line value?
        # But plot time line value is updated by us.
        
        # Let's just increment current_playback_time
        # But we need to sync it first if user scrubbed.
        # How to know if user scrubbed?
        # Maybe we can read the current value from plotter?
        current_plot_time = self.plotter_widget.time_line_fwd.value()
        
        # Calculate dt based on wall clock time
        now = time.time()
        real_dt = now - self.last_update_time
        self.last_update_time = now
        
        dt = real_dt * self.playback_speed
        next_time = current_plot_time + dt
        
        # Check bounds
        max_time = 0
        if self.video_timestamps_relative is not None:
            max_time = self.video_timestamps_relative[-1]
        elif self.fictrac_data and len(self.fictrac_data['timestamp']) > 0:
            max_time = self.fictrac_data['timestamp'][-1]
             
        if next_time > max_time:
            next_time = max_time
            self.pause_video()
            
        self.current_playback_time = next_time
        
        # Update everything
        self.plotter_widget.update_time_line(next_time) # This updates plot center
        self.on_plot_time_change(next_time) # This updates video frame

    def set_playback_speed(self, value):
        self.playback_speed = value / 10.0
        # print(f"Playback speed set to {self.playback_speed}x")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_file', type=str, help='Path to HDF5 stimulus file')
    parser.add_argument('--no-brain', action='store_true', help='Skip loading brain volume')
    parser.add_argument('--series', type=str, help='Stimulus series name (e.g. series_003)')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    viz = MultiModalVisualizer(no_brain=args.no_brain, series_name=args.series)
    
    if args.hdf5_file:
        import json
        import glob
        
        print(f"Auto-loading from HDF5: {args.hdf5_file}...")
        try:
            # Load Config
            config_path = os.path.join(os.getcwd(), 'config.json')
            config = {"patterns": {}}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # 1. Extract Date
            filename = os.path.basename(args.hdf5_file)
            date_str = "20251102" # Default fallback
            try:
                name_no_ext = os.path.splitext(filename)[0]
                date_str = name_no_ext.replace('-', '')
            except:
                pass
                
            # 2. Extract Series Number
            series_name = args.series
            series_num = "3" # Default
            series_3d = "003" # Default
            
            if series_name:
                try:
                    parts = series_name.split('_')
                    if len(parts) > 1 and parts[-1].isdigit():
                        series_3d = parts[-1]
                        series_num = str(int(series_3d))
                except:
                    pass
            
            # 3. Resolve Paths
            patterns = config.get('patterns', {})
            resolved_paths = {}
            
            for key in ['video', 'fictrac', 'brain', 'brain_xml', 'camera_timestamps']:
                if key in patterns:
                    raw_pattern = patterns[key]
                    try:
                        pattern = raw_pattern.format(date=date_str, series=series_num, series_3d=series_3d)
                    except KeyError:
                        pattern = raw_pattern
                    
                    if os.path.isabs(pattern):
                        search_pattern = pattern
                    else:
                        directory = os.path.dirname(args.hdf5_file)
                        search_pattern = os.path.join(directory, pattern)
                    
                    matches = glob.glob(search_pattern)
                    if matches:
                        resolved_paths[key] = matches[0]
                        print(f"Found {key}: {matches[0]}")
                    else:
                        print(f"Warning: Could not find {key} matching pattern: {pattern}")
            
            # 4. Load Data
            viz.load_data(
                hdf5_path=args.hdf5_file,
                series_name=args.series,
                video_path=resolved_paths.get('video'),
                fictrac_path=resolved_paths.get('fictrac'),
                brain_path=resolved_paths.get('brain'),
                brain_xml_path=resolved_paths.get('brain_xml'),
                cam_ts_path=resolved_paths.get('camera_timestamps')
            )
            
        except Exception as e:
            print(f"Failed to auto-load: {e}")
            import traceback
            traceback.print_exc()
        
    napari.run()

if __name__ == "__main__":
    main()
