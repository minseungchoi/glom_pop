from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSlider, QScrollArea, QComboBox, QLineEdit, QGridLayout, QGroupBox
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import cv2
import numpy as np
import json
import glob
import os
import h5py
from PyQt5.QtWidgets import QComboBox, QLineEdit, QGridLayout, QGroupBox

class DataLoader(QWidget):
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # HDF5 Selection
        self.layout.addWidget(QLabel("1. Select Stimulus File (HDF5)"))
        self.hdf5_layout = QHBoxLayout()
        self.edit_hdf5 = QLineEdit()
        self.btn_hdf5 = QPushButton("Browse")
        self.btn_hdf5.clicked.connect(self.browse_hdf5)
        self.hdf5_layout.addWidget(self.edit_hdf5)
        self.hdf5_layout.addWidget(self.btn_hdf5)
        self.layout.addLayout(self.hdf5_layout)
        
        # Series Selection
        self.layout.addWidget(QLabel("2. Select Series"))
        self.combo_series = QComboBox()
        self.combo_series.currentIndexChanged.connect(self.on_series_changed)
        self.layout.addWidget(self.combo_series)
        
        # Other Files (Auto-populated)
        self.files_group = QGroupBox("3. Associated Files")
        self.files_layout = QGridLayout()
        self.files_group.setLayout(self.files_layout)
        self.layout.addWidget(self.files_group)
        
        self.file_inputs = {}
        row = 0
        for key in ['video', 'fictrac', 'brain', 'brain_xml', 'camera_timestamps']:
            self.files_layout.addWidget(QLabel(f"{key.replace('_', ' ').title()}:"), row, 0)
            edit = QLineEdit()
            btn = QPushButton("...")
            btn.clicked.connect(lambda checked, k=key: self.browse_file(k))
            self.files_layout.addWidget(edit, row, 1)
            self.files_layout.addWidget(btn, row, 2)
            self.file_inputs[key] = edit
            row += 1
            
        # Load Button
        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self.load_data)
        self.layout.addWidget(self.btn_load)
        
        self.lbl_status = QLabel("Ready")
        self.layout.addWidget(self.lbl_status)
        
        # Load Config
        self.config = self.load_config()

    def load_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mmviz_dir = os.path.dirname(current_dir) # data is expected to be in the parent dir of mmviz pkg
        config_path = os.path.join(mmviz_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {"patterns": {}}

    def browse_hdf5(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select HDF5 File", "", "HDF5 Files (*.hdf5)")
        if filepath:
            self.edit_hdf5.setText(filepath)
            self.populate_series(filepath)
            self.auto_populate_files(filepath)

    def populate_series(self, hdf5_path):
        self.combo_series.clear()
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'Subjects' in f:
                    subjects_grp = f['Subjects']
                    # Assuming single subject or taking first
                    series_items = []
                    for subject_name in subjects_grp.keys():
                         if 'epoch_runs' in subjects_grp[subject_name]:
                            epoch_runs = subjects_grp[subject_name]['epoch_runs']
                            
                            for series_name in epoch_runs.keys():
                                protocol_name = "Unknown"
                                try:
                                    series_grp = epoch_runs[series_name]
                                    # Check for run_parameters where protocol info usually lives
                                    # VisAnalysis often puts it in run_parameters group attributes
                                    if 'protocol_ID' in series_grp.attrs:
                                        p_id = series_grp.attrs['protocol_ID']
                                        protocol_name = p_id.decode('utf-8') if isinstance(p_id, bytes) else p_id
                                except Exception:
                                    pass
                                
                                label = f"({series_name}) Subject {subject_name}: {protocol_name}"
                                series_items.append((label, series_name))
                    
                    # Sort by series name
                    series_items.sort(key=lambda x: x[1])
                    
                    for label, data in series_items:
                        self.combo_series.addItem(label, userData=data)
        except Exception as e:
            self.lbl_status.setText(f"Error reading HDF5: {str(e)}")

    def auto_populate_files(self, hdf5_path):
        # 1. Extract Date from HDF5 filename
        # Expected format: YYYY-MM-DD.hdf5 -> YYYYMMDD
        filename = os.path.basename(hdf5_path)
        date_str = "20251102" # Default fallback
        try:
            # Simple heuristic: remove extension, remove dashes
            name_no_ext = os.path.splitext(filename)[0]
            date_str = name_no_ext.replace('-', '')
        except:
            pass
            
        # 2. Extract Series Number
        series_name = self.combo_series.currentData()
        series_num = "3" # Default
        series_3d = "003" # Default
        
        if series_name:
            # Expected: series_003 or just "3"
            try:
                if str(series_name).isdigit():
                    series_num = str(series_name)
                    series_3d = f"{int(series_num):03d}"
                else:
                    parts = str(series_name).split('_')
                    if len(parts) > 1 and parts[-1].isdigit():
                        series_3d = parts[-1]
                        series_num = str(int(series_3d))
            except:
                pass
        
        patterns = self.config.get('patterns', {})
        
        for key, edit in self.file_inputs.items():
            if key in patterns:
                raw_pattern = patterns[key]
                # Format pattern
                try:
                    pattern = raw_pattern.format(date=date_str, series=series_num, series_3d=series_3d)
                except KeyError:
                    pattern = raw_pattern # Fallback if keys don't match
                
                # Search
                if os.path.isabs(pattern):
                    search_pattern = pattern
                else:
                    directory = os.path.dirname(hdf5_path)
                    search_pattern = os.path.join(directory, pattern)
                
                matches = glob.glob(search_pattern)
                if matches:
                    # Pick first match
                    edit.setText(matches[0])
                else:
                    # If no match, maybe just set the pattern as a hint?
                    # Or clear it. Let's clear it but print warning?
                    # Actually, user might want to see what it tried to find.
                    # But edit box expects a file.
                    edit.clear()
                    edit.setPlaceholderText(f"Not found: {os.path.basename(pattern)}")

    def browse_file(self, key):
        filepath, _ = QFileDialog.getOpenFileName(self, f"Select {key} File")
        if filepath:
            self.file_inputs[key].setText(filepath)

    def on_series_changed(self, index):
        # Update files when series changes
        hdf5_path = self.edit_hdf5.text()
        if hdf5_path:
            self.auto_populate_files(hdf5_path)

    def load_data(self):
        hdf5_path = self.edit_hdf5.text()
        series_name = self.combo_series.currentData()
        
        if not hdf5_path or not series_name:
            self.lbl_status.setText("Please select HDF5 file and Series.")
            return
            
        files = {k: v.text() for k, v in self.file_inputs.items()}
        
        self.lbl_status.setText("Loading...")
        try:
            self.visualizer.load_data(
                hdf5_path=hdf5_path,
                series_name=series_name,
                video_path=files.get('video'),
                fictrac_path=files.get('fictrac'),
                brain_path=files.get('brain'),
                brain_xml_path=files.get('brain_xml'),
                cam_ts_path=files.get('camera_timestamps')
            )
            self.lbl_status.setText("Data Loaded.")
        except Exception as e:
            self.lbl_status.setText(f"Error: {str(e)}")
            print(e)

    def update_paths(self, hdf5_path, series_name, file_paths):
        """Update the UI with the loaded file paths."""
        if hdf5_path:
            self.edit_hdf5.setText(hdf5_path)
            # We need to populate series combo to set the index correctly
            self.populate_series(hdf5_path)
        
        if series_name:
            index = self.combo_series.findData(series_name)
            
            # If not found, try formatted versions (e.g. '10' -> 'series_010')
            if index == -1:
                try:
                    s_str = str(series_name)
                    num = None
                    if s_str.isdigit():
                        num = int(s_str)
                    elif s_str.startswith('series_'):
                         parts = s_str.split('_')
                         if parts[-1].isdigit():
                            num = int(parts[-1])
                    
                    if num is not None:
                         candidate = f"series_{num:03d}"
                         index = self.combo_series.findData(candidate)
                except:
                    pass

            if index >= 0:
                self.combo_series.setCurrentIndex(index)
        
        for key, path in file_paths.items():
            if key in self.file_inputs and path:
                self.file_inputs[key].setText(path)
        
        self.lbl_status.setText("Data Loaded (Synced).")

class PlaybackControl(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        
        self.layout.addWidget(self.btn_play)
        self.layout.addWidget(self.btn_pause)
        self.layout.addWidget(self.btn_stop)
        
        # Speed Control
        self.layout.addWidget(QLabel("Speed:"))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 50) # 0.1x to 5.0x
        self.slider_speed.setValue(10)    # Default 1.0x
        self.slider_speed.setFixedWidth(150)
        self.slider_speed.valueChanged.connect(self.update_label)
        self.layout.addWidget(self.slider_speed)
        
        self.lbl_speed = QLabel("1.0x")
        self.layout.addWidget(self.lbl_speed)
        
    def update_label(self, value):
        speed = value / 10.0
        self.lbl_speed.setText(f"{speed:.1f}x")

class StimulusViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.lbl_info = QLabel("No Stimulus")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.lbl_info)
        self.scroll.setWidgetResizable(True)
        # self.scroll.setFixedHeight(150) # Removed fixed height to span whole tab
        self.layout.addWidget(self.scroll)

    def update_epoch(self, epoch):
        if epoch:
            idx_str = f" {epoch['index']}" if 'index' in epoch else ""
            params_str = f"Epoch{idx_str}: {epoch['name']}\n"
            for k, v in epoch['parameters'].items():
                if k not in ['name', 'epoch_unix_time', 'epoch_end_unix_time']:
                        params_str += f"{k}: {v}\n"
            self.lbl_info.setText(params_str)
        else:
            self.lbl_info.setText("No Stimulus")

class CustomViewBox(pg.ViewBox):
    def scaleBy(self, s=None, center=None, x=None, y=None):
        # Override to disable Y scaling via mouse (zoom)
        # s is the scale factor (sx, sy)
        if s is not None:
            if isinstance(s, (float, int)):
                sx, sy = s, s
            else:
                sx, sy = s[0], s[1]
            
            # Force Y scale to 1.0 (no change)
            s = (sx, 1.0)
        
        # Pass modified scale factor to super
        super().scaleBy(s, center, x, y)

class TimeSeriesPlotter(QWidget):
    timeChanged = pyqtSignal(float)
    sigEpochChanged = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Forward Velocity Plot Container
        self.fwd_container = QWidget()
        self.fwd_layout = QHBoxLayout()
        self.fwd_container.setLayout(self.fwd_layout)
        self.layout.addWidget(self.fwd_container)
        
        self.plot_fwd = pg.PlotWidget(title="Forward Velocity (mm/s)", viewBox=CustomViewBox())
        self.fwd_layout.addWidget(self.plot_fwd)
        self.plot_fwd.addLegend()
        self.time_line_fwd = pg.InfiniteLine(angle=90, movable=False, pen='r')
        # self.time_line_fwd.sigPositionChanged.connect(self.on_line_dragged) # No longer dragging line
        self.plot_fwd.addItem(self.time_line_fwd)
        # Enable mouse for both axes (pan works for both, zoom restricted by CustomViewBox)
        self.plot_fwd.setMouseEnabled(x=True, y=True)
        
        # Connect view change to update time (Center Cursor Mode)
        self.plot_fwd.getViewBox().sigXRangeChanged.connect(self.on_view_changed)
        
        # Zoom Slider
        self.slider_fwd_zoom = QSlider(Qt.Vertical)
        self.slider_fwd_zoom.setRange(1, 100)
        self.slider_fwd_zoom.setValue(100)
        self.slider_fwd_zoom.setToolTip("Y-Axis Zoom")
        self.slider_fwd_zoom.valueChanged.connect(self.update_fwd_zoom)
        self.fwd_layout.addWidget(self.slider_fwd_zoom)
        
        # Turning Velocity Plot Container
        self.turn_container = QWidget()
        self.turn_layout = QHBoxLayout()
        self.turn_container.setLayout(self.turn_layout)
        self.layout.addWidget(self.turn_container)

        self.plot_turn = pg.PlotWidget(title="Turning (left / CW) Velocity (deg/s)", viewBox=CustomViewBox())
        self.turn_layout.addWidget(self.plot_turn)
        self.plot_turn.addLegend()
        self.time_line_turn = pg.InfiniteLine(angle=90, movable=False, pen='r')
        # self.time_line_turn.sigPositionChanged.connect(self.on_line_dragged)
        self.plot_turn.addItem(self.time_line_turn)
        self.plot_turn.setMouseEnabled(x=True, y=True)
        
        # Zoom Slider Turn
        self.slider_turn_zoom = QSlider(Qt.Vertical)
        self.slider_turn_zoom.setRange(1, 100)
        self.slider_turn_zoom.setValue(100)
        self.slider_turn_zoom.setToolTip("Y-Axis Zoom")
        self.slider_turn_zoom.valueChanged.connect(self.update_turn_zoom)
        self.turn_layout.addWidget(self.slider_turn_zoom)
        
        # Link X-axis
        self.plot_turn.setXLink(self.plot_fwd)
        
    def plot_fictrac(self, time, fwd_vel, turning_vel, t0=0):
        self.plot_fwd.clear()
        self.plot_fwd.addItem(self.time_line_fwd)
        self.plot_turn.clear()
        self.plot_turn.addItem(self.time_line_turn)
        
        self.plot_fwd.plot(time-t0, fwd_vel, pen='g', name='Fwd Vel')
        self.plot_turn.plot(time-t0, turning_vel, pen='m', name='Turn Vel')
        
        # Store data limits for zooming
        self.fwd_max = np.max(fwd_vel) if len(fwd_vel) > 0 else 1.0
        self.fwd_min = np.min(fwd_vel) if len(fwd_vel) > 0 else 0.0
        
        self.turn_max = np.max(turning_vel) if len(turning_vel) > 0 else 1.0
        self.turn_min = np.min(turning_vel) if len(turning_vel) > 0 else 0.0
        
        # Handle outliers for initial view?
        # Let's just set slider to 100% (full range) initially
        self.slider_fwd_zoom.setValue(100)
        self.slider_turn_zoom.setValue(100)
        
        # Set default X-axis range to 60 seconds
        self.plot_fwd.setXRange(0, 60, padding=0)

    def update_fwd_zoom(self, value):
        if hasattr(self, 'fwd_max'):
            # Calculate target span based on slider
            data_span = self.fwd_max - self.fwd_min
            if data_span == 0: data_span = 1
            
            factor = (value / 100.0) ** 2 
            target_span = data_span * factor
            
            # Get current view center to preserve panning
            current_range = self.plot_fwd.viewRange()[1]
            current_center = (current_range[0] + current_range[1]) / 2
            
            # Set new range centered on current view
            self.plot_fwd.setYRange(current_center - target_span/2, current_center + target_span/2, padding=0)

    def update_turn_zoom(self, value):
        if hasattr(self, 'turn_max'):
            data_span = self.turn_max - self.turn_min
            if data_span == 0: data_span = 1
            
            factor = (value / 100.0) ** 2 
            target_span = data_span * factor
            
            current_range = self.plot_turn.viewRange()[1]
            current_center = (current_range[0] + current_range[1]) / 2
            
            self.plot_turn.setYRange(current_center - target_span/2, current_center + target_span/2, padding=0)

    def on_view_changed(self):
        # When user pans/zooms X, update time to center of view
        if getattr(self, 'programmatic_scroll', False):
            return
            
        view_range = self.plot_fwd.viewRange()[0]
        center_x = (view_range[0] + view_range[1]) / 2
        
        # Update lines
        self.time_line_fwd.setValue(center_x)
        self.time_line_turn.setValue(center_x)
        
        # Emit signal
        self.timeChanged.emit(center_x)
        self.update_stim_label(center_x)

    def on_line_dragged(self, line):
        # Deprecated in Center Cursor Mode
        pass

    def update_stim_label(self, time):
        # Update stimulus info label
        if hasattr(self, 'epochs'):
            current_epoch = None
            for epoch in self.epochs:
                if epoch['start'] <= time <= epoch['end']:
                    current_epoch = epoch
                    break
            
            self.sigEpochChanged.emit(current_epoch)

    def plot_stimulus(self, epochs):
        # epochs is a list of dicts: {'start': t_start, 'end': t_end, 'name': name, 'parameters': dict}
        self.epochs = epochs # Store for lookup
        for epoch in epochs:
            # Calculate shade region based on pre_time and stim_time if available
            start_t = epoch['start']
            end_t = epoch['end']
            
            params = epoch['parameters']
            if 'pre_time' in params and 'stim_time' in params:
                try:
                    pre_t = float(params['pre_time'])
                    stim_t = float(params['stim_time'])
                    
                    # Shade only the stim_time part
                    shade_start = start_t + pre_t
                    shade_end = shade_start + stim_t
                except ValueError:
                    # Fallback if conversion fails
                    shade_start = start_t
                    shade_end = end_t
            else:
                # Fallback if params missing
                shade_start = start_t
                shade_end = end_t
        
            # Add region to both plots
            # Use a light gray/blue shading, no boundary lines
            brush = (200, 200, 255, 50)
            region_fwd = pg.LinearRegionItem([shade_start, shade_end], brush=brush, movable=False)
            for line in region_fwd.lines: line.setPen(pg.mkPen(None))
            self.plot_fwd.addItem(region_fwd)
            
            region_turn = pg.LinearRegionItem([shade_start, shade_end], brush=brush, movable=False)
            for line in region_turn.lines: line.setPen(pg.mkPen(None))
            self.plot_turn.addItem(region_turn)
            
            # Add text label for name (only on top plot)
            text = pg.TextItem(epoch['name'], anchor=(0, 1), color=(200, 200, 255))
            text.setPos(start_t, 0) # Keep label at actual start of epoch
            self.plot_fwd.addItem(text)
        
    def update_time_line(self, time):
        # Called when video plays or external time change
        self.programmatic_scroll = True
        
        self.time_line_fwd.setValue(time)
        self.time_line_turn.setValue(time)
        
        # Center view on time
        current_range = self.plot_fwd.viewRange()[0]
        half_width = (current_range[1] - current_range[0]) / 2
        self.plot_fwd.setXRange(time - half_width, time + half_width, padding=0)
        
        self.update_stim_label(time)
        
        self.programmatic_scroll = False

class VideoViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.layout.addWidget(self.image_view)
        
        self.cap = None
        
    def load_video(self, filepath):
        # We could use pims or cv2. 
        # For random access, cv2 with set(CAP_PROP_POS_MSEC) might be okay, 
        # or loading whole video into memory if small enough.
        # Given "cam_18384364.mp4", let's try pims for lazy loading or just cv2.
        self.cap = cv2.VideoCapture(filepath)
        
        # Show first frame
        ret, frame = self.cap.read()
        if ret:
            # CV2 is BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Transpose for pyqtgraph (col, row, channel)
            frame = np.transpose(frame, (1, 0, 2))
            self.image_view.setImage(frame)

    def clear_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.image_view.clear()

    def set_time(self, time_sec):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.transpose(frame, (1, 0, 2))
                self.image_view.setImage(frame)

    def set_frame(self, frame_idx):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.transpose(frame, (1, 0, 2))
                self.image_view.setImage(frame)
