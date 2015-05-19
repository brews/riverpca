#! /usr/bin/env python3

# Copyright 2014 S. B. Malevich <malevich@email.arizona.edu>
# 2014-02-16

# Create a PDF atlas of maps showing 20th Century reanalysis search result plots 
# for 700, 500 and 250 mb through NDJ and FMA.

# THIS IS OUTDATED!

import numpy as np
import pandas as pd
import datetime
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

FIELD_PATH = "./data/20th_rean_V2.npz"
PDF_PATH = "./plots/atlas_high_20threanv2.pdf"
SEARCH_PATH = "./data/search_high_20threanv2.tsv"

def main():
    search_raw = pd.read_table(SEARCH_PATH, sep = "\t")
    data_raw = np.load(FIELD_PATH)

    print("Calculating anomalies...")  # DEBUG
    hgt_minmax = {"ndj": {700: (None, None),
                          500: (None, None), 
                          250: (None, None),},
                  "fma": {700: (None, None), 
                          500: (None, None), 
                          250: (None, None),}}
    hgt_anomaly = np.empty(data_raw["data"].shape)
    for i in range(len(search_raw["season"].unique())):
        s = search_raw["season"].unique()[i]
        for j in range(len(search_raw["height"].unique())):
            h = search_raw["height"].unique()[j]
            for k in range(len(data_raw["time"])):
                zonal_mean = data_raw["data"][i, j, k].mean(axis = 1)[:, np.newaxis]
                hgt_anomaly[i, j, k] = data_raw["data"][i, j, k] - zonal_mean
            hgt_minmax[s][h] = (hgt_anomaly[i, j].min(), hgt_anomaly[i, j].max())

    try:
        print("Plotting...")  # DEBUG
        pdf = PdfPages(PDF_PATH)
        for k in tqdm(range(len(data_raw["time"]))):
            t = data_raw["time"][k]
            fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (8, 10), dpi = 300)
            for i in range(len(search_raw["season"].unique())):
                s = search_raw["season"].unique()[i]
                for j in range(len(search_raw["height"].unique())):
                    h = search_raw["height"].unique()[j]
                    search_target = search_raw.query("height==" + str(h) + " & season=='" + s + "' & year==" + str(t))
                    m = Basemap(ax = axes[j, i], width=12000000,height=9000000,rsphere=(6378137.00,6356752.3142),resolution='c',area_thresh=1000,projection='lcc',lat_1=45,lat_2=55,lat_0=50,lon_0=-160)
                    m.drawcoastlines(color = "#696969")
                    m.drawparallels(np.arange(-90, 110, 20), color = "#696969")
                    m.drawmeridians(np.arange(0, 360, 60), color = "#696969")
                    ct1 = m.contour(data_raw["lon"], data_raw["lat"], hgt_anomaly[i, j, k], latlon = True, 
                                    linewidths = 0.5, colors = "k", rasterized = True)
                    ct2 = m.pcolor(data_raw["lon"], data_raw["lat"], hgt_anomaly[i, j, k], latlon = True, cmap = plt.cm.RdBu_r, vmin = hgt_minmax[s][h][0], vmax = hgt_minmax[s][h][1], rasterized = True)
                    cb = m.colorbar(ct2, "right", size = "5%", pad = "2%")
                    m.scatter(search_target["lon"].tolist(), search_target["lat"].tolist(), marker = 'o', latlon = True, facecolors = "none", edgecolors='r', rasterized = True)
                    axes[j, i].set_title(str(t) + "-" + s + "-" + str(h) + "mb")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches = 0)
            plt.close()
        d = pdf.infodict()
        d["Title"] = "20th Century Reanalysis V2 High Pressure search atlas"
        d["Author"] = "S. Brewster Malevich"
        d["CreationDate"] = datetime.datetime.today()
    finally:
        print("Closing PDF...")  # DEBUG
        pdf.close()

if __name__ == '__main__':
    main()