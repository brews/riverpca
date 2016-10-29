#! /usr/bin/env python3

# Download and parse data needed for analysis.

import os
import parse


def main():
    # Need to make this empty directory for figure export by notebooks.
    plots_dir = './plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # These may take some time to download and run.
    parse.tcr(outpath = './data/tcrv2_z500_season.nc')
    parse.ersst(outpath = './data/ersstv3b_season.nc')
    parse.hcdn(outpath = './data/stationdb.sqlite')


if __name__ == '__main__':
    main()
