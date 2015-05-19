#! /usr/bin/env python3

# Copyright 2015 S. B. Malevich <malevich@email.arizona.edu>
# 2014-03-07

# Animate a North Pacific map showing a grid of dynamic agents converge on their solution using 20th Century Renalysis V2.
# The PNG files that are output to ANIMATIONPATH can be turned into a nice GIF with the bash command:
#    convert -delay 20 -loop 0 *.png grid_search_animation.gif
# when in that directory.

import numpy as np
import pylab as plt
from mpl_toolkits.basemap import Basemap
import dynamic

TWENTYPATH = "./data/20th_rean_V2.npz"
ANIMATIONPATH = "./plots/grid_search_animation/"


rean = np.load(TWENTYPATH)

target = rean["data"][0, 2][rean["time"] == 1977]  # Should be NDJ at 250 mb for 1977.
eddy = target - target.mean(2)[:, :, np.newaxis]

problem = dynamic.Problem()
problem.set_cube(eddy, lat = rean["lat"], lon = rean["lon"], lat_index = 1, lon_index = 2, time_index = 0)
sample_lon, sample_lat = np.meshgrid(np.arange(130, 254, 4), np.arange(32, 84, 4))
sample_grid = np.rec.fromarrays([sample_lat, sample_lon]).flatten()
for c in sample_grid:
    problem.add_agent(c, "up")
    problem.add_agent(c, "down")
problem.run()
sol = problem.get_solution()


# plt.figure(figsize = (8, 10))
# m = Basemap(resolution = "c", projection = "ortho", lat_0 = 60, lon_0 = -170)
# m.drawcoastlines(color = "#696969")
# m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
# m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
# m.contour(rean["lon"], rean["lat"], eddy[0], latlon = True, linewidths = 0.5, colors = "k", rasterized = True)
# ct1 = m.pcolor(rean["lon"], rean["lat"], eddy[0], latlon = True, cmap = plt.cm.RdBu_r, rasterized = True)
# final_lat = []; final_lon = []
# for a in range(len(sol)):
#     la = []; lo = []
#     for s in range(len(sol[a][0])):
#         latlon = problem.data.yx2latlon(sol[a][0][s][0][1:3])
#         la.append(latlon[0])
#         lo.append(latlon[1])
#     final_lat.append(la[-1])
#     final_lon.append(lo[-1])
# m.scatter(sample_lon, sample_lat, marker = ".", latlon = True, rasterized = True)
# cb = m.colorbar(ct1, "bottom", size = "5%", pad = "2%")
# plt.show()


# plt.figure(figsize = (8, 10))
# m = Basemap(resolution = "c", projection = "ortho", lat_0 = 60, lon_0 = -170)
# m.drawcoastlines(color = "#696969")
# m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
# m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
# m.contour(rean["lon"], rean["lat"], eddy[0], latlon = True, linewidths = 0.5, colors = "k", rasterized = True)
# ct1 = m.pcolor(rean["lon"], rean["lat"], eddy[0], latlon = True, cmap = plt.cm.RdBu_r, rasterized = True)
# final_lat = []; final_lon = []
# for a in range(len(sol)):
#     la = []; lo = []
#     for s in range(len(sol[a][0])):
#         latlon = problem.data.yx2latlon(sol[a][0][s][0][1:3])
#         la.append(latlon[0])
#         lo.append(latlon[1])
#     m.plot(lo, la, linewidth = 0.5, latlon = True, color = "k", alpha = 0.2)
#     final_lat.append(la[-1])
#     final_lon.append(lo[-1])
# m.scatter(final_lon, final_lat, marker = "o", latlon = True, facecolor = "k", rasterized = True)
# cb = m.colorbar(ct1, "bottom", size = "5%", pad = "2%")
# plt.show()


maxsteps = np.max([len(x[0]) for x in sol])
jitter_noise = np.zeros(len(sol))
jitter_noise += np.random.randn(len(sol)) * 0.75
for st in range(maxsteps):
    plt.figure(figsize = (8, 10))
    m = Basemap(projection='npstere',boundinglat=20,lon_0=210,resolution='c')
    # m = Basemap(resolution = "c", projection = "ortho", lat_0 = 60, lon_0 = -170)
    m.drawcoastlines(color = "#696969")
    m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
    m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
    m.contour(rean["lon"], rean["lat"], eddy[0], latlon = True, linewidths = 0.5, colors = "k", rasterized = True)
    ct1 = m.contourf(rean["lon"], rean["lat"], eddy[0], latlon = True, cmap = plt.cm.RdBu_r, rasterized = True)
    final_lat = []; final_lon = []
    for a in range(len(sol)):
        la = []; lo = []
        for sm in range(st + 1):
            agent_max = len(sol[a][0]) - 1
            s = sm
            if sm > agent_max:
                s = agent_max
            latlon = problem.data.yx2latlon(sol[a][0][s][0][1:3])
            la.append(latlon[0])
            lo.append(latlon[1])
        final_lat.append(la[-1])
        final_lon.append(lo[-1])
        # m.plot(lo + jitter_noise[a], la + jitter_noise[::-1][a], linewidth = 0.5, latlon = True, color = "k", alpha = 0.05)
    m.scatter(final_lon + jitter_noise, final_lat + jitter_noise[::-1], marker = "o", latlon = True, facecolors = "none", rasterized = True, alpha = 0.75)
    cb = m.colorbar(ct1, "bottom", size = "5%", pad = "2%")
    cb.set_label("m")
    plt.title("20th Century Rean. V2 - 1977 NDJ 250 mb zonal anomaly\nStep " + str(st))
    # plt.show()
    plt.tight_layout()
    plt.savefig(ANIMATIONPATH + str(st).zfill(2) + ".png")
    plt.close()
