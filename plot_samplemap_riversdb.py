#! /usr/bin/env python
# Copyright 2014 S. Brewster Malevich <malevich@email.arizona.edu>
# 2014-11-05
# Sample map of rivers for my AMS 2014 presentation. This is a map from a 
# subset of the same data for an NSF proposal.

import sqlite3
import numpy as np
import pylab as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.patheffects as PathEffects

DB_PATH = "./data/riversdb.sqlite"
MAP_PATH = "./plots/samplemap_riversdb.png"

def nl_insert(s, i):
    """Insert a newline into string s, at whitespace index i."""
    l = s.split(" ")
    l.insert(i, "\n")
    s = " ".join(l)
    return s

def main():
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute("SELECT lat, lon, gaugename FROM GaugeMeta WHERE gaugename NOT LIKE ('Green River near Green River, WY')")
    output = c.fetchall()
    con.close()
    lat = [t[0] for t in output]
    lon = [t[1] for t in output]
    labs = [t[2] for t in output]
    # labs = [i.split(" ") for i in labs]
    # for i in labs: i.insert(2, "\n ")
    # labs = [" ".join(i) for i in labs]
    labs[0] = nl_insert(labs[0], 2)
    labs[1] = nl_insert(labs[1], 2)
    labs[2] = nl_insert(labs[2], 2)
    labs[3] = nl_insert(labs[3], 5)
    labs[4] = nl_insert(labs[4], 3)
    labs[5] = nl_insert(labs[5], 2)
    labs[6] = nl_insert(labs[6], 4)
    labs[7] = nl_insert(labs[7], 3)
    labs[8] = nl_insert(labs[8], 2)
    labs[9] = nl_insert(labs[9], 2)
    labs[10] = nl_insert(labs[10], 4)
    labs[11] = nl_insert(labs[11], 4)
    labs[12] = nl_insert(labs[12], 2)

    lat_labs = lat[:]
    lon_labs = lon[:]
    lat_labs[0] -= 0.35
    lon_labs[0] -= 6.3
    lon_labs[1] -= 6
    lat_labs[2] += 0.35
    lon_labs[2] += 0.5
    lat_labs[3] -= 1.5
    lon_labs[3] -= 1
    lat_labs[4] -= 0.1
    lon_labs[4] -= 3
    lat_labs[5] -= 1.6
    lon_labs[5] -= 3.5
    lat_labs[6] += 0.35
    lon_labs[6] -= 7
    lat_labs[7] -= 0.4
    lon_labs[7] += 0.5
    lat_labs[8] -= 1.5
    lon_labs[8] += 0.2
    lat_labs[9] += 0.5
    lon_labs[9] -= 3.5
    lon_labs[10] += 0.2
    lat_labs[11] += 0.27
    lon_labs[12] -= 1.5
    lat_labs[12] += 0.37

    plt.figure(figsize = (10, 8))
    m = Basemap(width = 2500000, height = 2300000, 
                resolution = 'l', projection = 'stere', 
                lat_ts = 40.0, 
                lat_0 = 40.0, lon_0 = -112.0)
    # m.drawmapboundary()
    m.drawcoastlines(color = "#333333")
    m.drawstates(linewidth = 0.7, color = "#333333")
    m.drawcountries(color = "#333333")
    m.shadedrelief()
    parallels = np.arange(0., 81, 5)
    m.drawparallels(parallels, labels = [True, False, False, True], color = "#333333")
    meridians = np.arange(10., 351., 5)
    m.drawmeridians(meridians, labels = [False, True, False, True], color = "#333333")
    xpt,ypt = m(lon, lat)
    xlpt,ylpt = m(lon_labs, lat_labs)    
    m.scatter(xpt, ypt, s = 37, marker = 'o')
    for i in range(len(labs)):
        plt.text(xlpt[i], ylpt[i], labs[i], path_effects = [PathEffects.withStroke(linewidth = 3.5, foreground = "w")], color = "#333333")
    plt.savefig(MAP_PATH, bbox_inches = 0, dpi = 300)

if __name__ == '__main__':
    main()