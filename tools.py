# Copyright 2015 S. B. Malevich <malevich@email.arizona.edu>
# 2015-01-09

# Random collections of often-used code for the project.

import sqlite3
from calendar import monthrange
from numpy.linalg import svd
import scipy.stats as stats
import pylab as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


def check_monthly(dbpath, yearlow, yearhigh, westoflon=-104, eastoflon=-125):
    """Test that each of the USGS stations in the sqllite database have complete observations over the period we are interested in"""
    expected_obs_no = 12 * ((yearhigh - yearlow) + 1)
    candidate_pass = []
    conn = sqlite3.connect(dbpath)
    c = conn.cursor()
    c.execute("SELECT stationid FROM StationInfo WHERE longage < ? AND longage > ?", (westoflon, eastoflon))
    candidate_ids = c.fetchall()
    for station in candidate_ids:
        c.execute("SELECT month, year, count FROM StationMonthly WHERE stationid=? AND year >= ? AND year <= ?", (station[0], yearlow, yearhigh))
        myc = c.fetchall()
        # Check that we have all months:
        if len(myc) != expected_obs_no:
            continue
        # Check that no months are missing days:
        exam = [False] * expected_obs_no
        for i in range(expected_obs_no):
            if monthrange(myc[i][1], myc[i][0])[1] == myc[i][2]:
                exam[i] = True
        if not all(exam):
            continue
        candidate_pass.append(station[0])
    conn.close()
    return candidate_pass

def spigamma(x):
    """Transform data like SPI after fitting a gamma function."""
    zero_mask = x == 0
    q = zero_mask.sum()/len(x)
    g_shape, g_loc, g_scale = stats.gamma.fit(x[~zero_mask], floc = 0)
    g_fit = q + (1 - q) * stats.gamma.cdf(x, a = g_shape, scale = g_scale)
    return stats.norm.ppf(g_fit)

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    """ Varimax rotation of a 'loadings matrix', Phi. Output is (rotated loadings, rotation matrix). This is loosely based on the R stats library's varimax()."""
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        b = np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, diag(diag(np.dot(Lambda.T,Lambda)))))
        u,s,vh = svd(b)
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol:
            break
    return (np.dot(Phi, R), R)

def map_gages(lat, lon):
    """Create a simple sample map of USGS gages in the Western US, given lat/lon"""
    f = plt.figure(figsize = (4, 6))
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
    plt.title("n = " + str(len(lon)))
    return f
