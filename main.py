#! /usr/bin/env python3

import parse

def main():
    # These may take some time to download and run.
    parse.tcr(outpath = './data/tcrv2_z500_season.nc')
    parse.ersst(outpath = './data/ersstv3b_season.nc')
    parse.hcdn(outpath = './data/stationdb.sqlite', 
               inpath = './data/HCDN-2009_Station_Info.tsv')

if __name__ == '__main__':
    main()
