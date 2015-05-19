#! /usr/bin/env python3
# 2015-01-09

# This script creates a number of PNG files in the MAP_PATH directory. We can then run `convert -delay 100 -loop 0 *.png usgs_decayanimation.gif` to create a gif.

import sqlite3
from calendar import monthrange
import numpy as np
import pandas as pd
import pylab as plt
from mpl_toolkits.basemap import Basemap
from tools import check_monthly

DB_PATH = "./data/stationdb.sqlite"
MAP_PATH = "./plots/usgssampledecay/"
YEAR_HIGH = 2011

def plot_map(dbpath, mappath, yearlow, yearhigh):
    target_stations = [i for i in check_monthly(dbpath, yearlow, yearhigh)]
    conn = sqlite3.connect(dbpath)
    sql_query = "SELECT stationid, latgage, longage FROM StationInfo WHERE stationid IN ({seq}) ORDER BY stationid ASC".format(seq = ",".join(["?"] * len(target_stations)))
    latlon = pd.read_sql(sql_query, conn, params = target_stations)
    conn.close()
    lon, lat = latlon["longage"].as_matrix(), latlon["latgage"].as_matrix()


    plt.figure(figsize = (4, 6))
    m = Basemap(width = 2000000, height = 2300000, 
                resolution = 'l', projection = 'stere', 
                lat_ts = 40.0, 
                lat_0 = 40.0, lon_0 = -114.0)
    # m.drawmapboundary()
    m.drawcoastlines(color = "#333333")
    m.drawstates(linewidth = 0.7, color = "#333333")
    m.drawcountries(color = "#333333")
    m.shadedrelief()
    m.scatter(lon, lat, s = 30, marker = 'o', latlon = True, facecolors = "none", edgecolors='r')
    plt.title(str(yearhigh) + "-" + str(yearlow) + "(n = " + str(len(target_stations)) + ")")
    plt.savefig(mappath, bbox_inches = 0, dpi = 150)

def main():
    for lowy in range(1995, 1905, -5):
        print(lowy)  # DEBUG
        outpath = MAP_PATH + "samplemap_usgs_" + str(YEAR_HIGH) + "-" + str(lowy) + ".png"
        plot_map(DB_PATH, outpath, lowy, YEAR_HIGH)

if __name__ == '__main__':
    main()