# riverpca

Climate-oriented PCA of USGS river gages in the western US. This code was part of analysis and data collection for a paper currently in review.

Analysis for the paper is in the `notebooks/` directory. This directory contains a number of Jupyter notebooks. You can easily view the code and output figures in these notebooks online through GitHub, without the need to download data, scripts, or code. The other files and directories in this repository are support material used for the analysis.

## Files and directories

* `data/`: Contains data that is output by the project scripts or downloaded.

* `notebooks/`: Jupyter notebooks for streamflow PCA covering different periods within the past century. You can easily view these notebooks online by browsing to them in GitHub.

* `parse/`: Python module to download and parse the data needed for analysis. Has tools to download and parse TCRv2 and ERSST fields. It also has tools to parse the HCDN-2009 metadata and mine USGS Water Services for the HCDN gages. This module is used by `main.py`.

* `tools/`: Specialized plotting and statistical tools of varying use and quality. These are used by the jupyter notebooks in `notebooks/`.

* `main.py`: Main script to download, parse, and clean the data needed for this projects analysis.

* `requirements.txt`: Computer readable list of libraries and modules to recreate the coding environment used for this analysis. If you use the Conda Python distribution or Miniconda package manager, you can use this file to setup for tweak or recreate our analysis.


The modules `parse/` and `tools/` contain unit tests that can be run with `pytest tools` or `pytest parse`, if you have the pytest library installed. These are more very loose sanity check and not comprehensive tests.
