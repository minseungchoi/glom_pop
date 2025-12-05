# Multimodal Data Visualizer (mmviz)

A Python package for visualizing multimodal data (imaging, behavior, stimulus).

## Installation

```bash
pip install -e .
```

## Usage

Run the visualizer from the command line:

```bash
mmviz --hdf5_file /path/to/data.hdf5 --series series_003
```

Or run as a module:

```bash
python -m mmviz.visualizer --hdf5_file ...
```

## Structure

- `mmviz/visualizer.py`: Main application logic.
- `mmviz/widgets.py`: GUI widgets (Napari, PyQtGraph).
- `mmviz/utils.py`: Utility functions (metadata extraction).
