#! /usr/bin/env python3
# 2015-02-26

# A basemap plot of USGS stations.

import sqlite3
from calendar import monthrange
import numpy as np
import pandas as pd
import pylab as plt
from mpl_toolkits.basemap import Basemap
from tools import check_monthly

DB_PATH = "./data/stationdb.sqlite"
MAP_PATH = "./plots/samplemap_usgs_long.png"
YEAR_LOW = 1949
YEAR_HIGH = 2011

def main():
    target_stations = [i for i in check_monthly(DB_PATH, YEAR_LOW, YEAR_HIGH)]
    conn = sqlite3.connect(DB_PATH)
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
    parallels = np.arange(0., 81, 10)
    m.drawparallels(parallels, labels = [True, False, True, False], color = "#333333")
    meridians = np.arange(10., 351., 10)
    m.drawmeridians(meridians, labels = [False, True, False, True], color = "#333333")
    m.scatter(lon, lat, s = 30, marker = 'o', latlon = True, facecolors = "none", edgecolors='r')
    plt.title("n = " + str(len(target_stations)))
    plt.savefig(MAP_PATH, bbox_inches = 0, dpi = 300)

if __name__ == '__main__':
    main()