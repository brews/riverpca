#! /usr/bin/env bash
# 2014-02-27

# Note there is a setup python script files in ./data/ERA-I/. This downloads big data so don't use it unless you have to.

./create_usgsdb.py
./create_riversdb.py

./parse_20threanv2.py
./parse_erai.py
./parse_ersstv3.py
./parse_noaaoiv2.py

./riverEOFS_short.py
./riverEOFS_long.py

./create_search_20threanv2_coastal.py
./create_search_20threanv2_pacificgrid.py


./plot_samplemap_riversdb.py
./plot_samplemap_usgs_short.py
./plot_samplemap_usgs_long.py
./plot_samplemap_usgs_sampledecay.py
# ./plot_20threanv2_atlas.py  # Plots big PDF.
./combine_river_search.py