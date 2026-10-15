# Bias Correction Tool (GUI)

A PyQt-based desktop application designed to apply absolute and optional relative bias adjustments to coastal storm hydrograph simulations (ADCIRC and STWAVE), with optional hourly resampling centered around peak surge.

## Prerequisites

   Python 3.x
   h5py>=3.16.0
   numpy
   PyQt6>=6.11.0
   qtpy>=2.4.3
   scipy>=1.15.3


## Installation

1.  Clone the repository or download the source files.
2.  Install the required Python packages using the provided `requirements.txt` file
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Launch the graphical interface from the command line
```bash
python bias_correction.py
