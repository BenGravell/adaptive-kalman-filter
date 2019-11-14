# Adaptive Kalman filter

The code in this repository implements an adaptive Kalman filter for linear systems with unknown process and measurement noise covariance.

The filter is based on the work "Adaptive Kalman Filter for Detectable Linear Time-Invariant Systems" by Moghe, Zanetti and Akella at https://arc.aiaa.org/doi/full/10.2514/1.G004359.

* Demo notebook (runs in-browser, no installation required)
  * [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BenGravell/adaptive_kalman_filter/master?filepath=adaptive_kalman_filter_scalar_tiny_plot2.ipynb)
  * [Precomputed version](https://nbviewer.jupyter.org/github/BenGravell/adaptive_kalman_filter/blob/master/adaptive_kalman_filter_scalar_tiny_plot2_precomp.ipynb)

## Package dependencies
* NumPy
* SciPy
* Matplotlib
* namedlist
* RISE (for notebooks in slide presentation mode)
