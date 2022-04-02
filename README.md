# KDE2B
Tool for calculating Gaussian Kernel Density Estimations (KDEs) on 2D bounded data sets.

Conventional KDEs usually do not deal well with bounded data, i.e. when data points are concentrated near the edges of the defined space. This little tool provides different boundary correction methods to ensure the data set is reasonably well represented by the KDE.

## Installation
This is kept very simple on purpose. No installation necessary; just put gaussian_kde.py in the directory where you want to use it, and import it to your Python project with:
```
import gaussian_kde as gkde
```

## Usage
It's probably easiest to pick either `1D_KDE.py` or `2D_KDE.py` from the examples directory, and start from there. The scripts demonstrate how to implement a KDE in one or two dimensions, with and without boundary corrections. Two example images show a comparison of the different methods. Remember you need to have the example script and the `gaussian_kde.py` in the same directory.

If you need more information, the `gaussian_kde.py` script features  extensive docstrings.
