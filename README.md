# Particle_Size_Distribution_Log_Fit_Python

**Version 0.1.0**

**Summary:**
This is the initial release of the Python adaptation of the Particle Number Size Distribution (PNSD) fitting module, DOFIT, originally developed by Tareq Hussein. The Particle_Size_Distribution_Log_Fit_Python is a log-normal fitting module designed to identify the best number of modes based on overlapping modes and RMSE for any given PNSD. At present, it supports up to 4 modes only.

![alt text](https://github.com/theodorekhadir/dofit_python/blob/master/test.png)

**Features:**

Translated from MATLAB to Python, enabling more researchers to access and use the module.
Algorithm selects the best number of modes based on overlapping modes and RMSE for a given PNSD.
Returns a set of log-normal parameters for each fitted mode.
Supports up to 4 modes.

**Known Limitations:**

Currently, the module is only able to handle less than 4 modes. We plan to extend this in future updates.
This version does not include multiprocessing capabilities.
Planned Updates:

**In the next versions**

We plan to incorporate multiprocessing to speed up the processing time and improve the efficiency of the module. One of our upcoming goals is to package this module for distribution through Python's package installer, pip. This will make the installation process more straightforward and enable seamless integration of this module into your Python projects.

**How to cite?**

If you use this module in your research, please cite it. A Digital Object Identifier (DOI) has been generated for this module, which you can use for citation purposes in academic work: 

*DOI (version 0.1.0)*: https://doi.org/10.5281/zenodo.8017043

**Acknowledgements:**

This module is an adaptation of the original DOFIT module developed by Tareq Hussein. The original module can be accessed here: http://www.borenv.net/BER/archive/pdfs/ber10/ber10-337.pdf

*If you want to contribute to this module or have any question, don't hesitate to email me:
theodore.khadir@aces.su.se*
