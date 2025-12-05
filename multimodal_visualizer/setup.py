from setuptools import setup, find_packages

setup(
    name='mmviz',
    version='0.1.0',
    description='Multimodal Data Visualizer',
    author='Minseung Choi',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'h5py',
        'napari',
        'PyQt5',
        'pyqtgraph',
        'scipy',
        'nibabel',
        'visanalysis', # Assuming this is installed
        'opencv-python',
    ],
    entry_points={
        'console_scripts': [
            'mmviz=mmviz.visualizer:main',
        ],
    },
)
