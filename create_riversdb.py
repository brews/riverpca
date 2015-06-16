#! /usr/bin/env python3
# Create an SQLite3 database containing river gage data, reconstructions
# and metadata. We're also carefully cleaning the data here.

import sqlite3 as sql
import pandas as pd
import numpy as np
from calendar import monthrange

DATACACHE = "./data/rivers/"
DB_PATH = "./data/riversdb.sqlite"
SCHEMA_STRING = """
            DROP TABLE IF EXISTS GageMeta;
            CREATE TABLE GageMeta(id INTEGER PRIMARY KEY,
                                   gagename TEXT,
                                   basin TEXT,
                                   river TEXT,
                                   units TEXT,
                                   lat REAL,
                                   lon REAL,
                                   citation TEXT,
                                   notes TEXT,
                                   url TEXT,
                                   datecited TEXT);
            DROP TABLE IF EXISTS Gage;
            CREATE TABLE Gage(id INTEGER PRIMARY KEY,
                               gagename TEXT,
                               year INTEGER,
                               value REAL,
                               FOREIGN KEY (gagename) REFERENCES GageMeta(gagename));
            DROP TABLE IF EXISTS ReconMeta;
            CREATE TABLE ReconMeta(id INTEGER PRIMARY KEY,
                                   reconname TEXT,
                                   units TEXT,
                                   target TEXT,
                                   citation TEXT,
                                   notes TEXT,
                                   url TEXT,
                                   datecited TEXT,
                                   FOREIGN KEY (target) REFERENCES GageMeta(gagename));
            DROP TABLE IF EXISTS Recon;
            CREATE TABLE Recon(id INTEGER PRIMARY KEY,
                               reconname TEXT,
                               year INTEGER,
                               value REAL,
                               FOREIGN KEY (reconname) REFERENCES ReconMeta(reconname));
                """

def seconds_in_month(date=None, year=None, month=None):
    """Return the number of seconds in a given month"""
    try:
        year = date.year
        month = date.month
    except AttributeError:
        pass
    seconds_in_day = 24 * 60 * 60
    n_days = monthrange(year, month)[-1]
    return n_days * seconds_in_day

def create_database(path, data, schema):
    """Create an sqlite3 database of river and reconstruction information.

    Args:
        path: Path to existing or desired sqlite3 database.
        data: A dictionary with a key for each table which is to be written 
            to. Each item in the dictionary is a list with a tuple for each 
            values that is to be entered into the table. It is assumed that 
            the first item of each table is an integer primary key.
        schema: A string SQLite3 script that defines initializes the database, 
            creating any tables needed.

    Returns:
        Nothing is returned, an sqlite3 database produced at `path` is the side effect.

    Raises:
        Nothing.
    """
    with sql.connect(path) as con:
        cur = con.cursor()
        cur.executescript(schema)
        for i in data.keys():
            head = "INSERT INTO " + i + " VALUES "
            variables = "(NULL, " + " ?," * (len(data[i][0]) - 1) + " ?)"
            cur.executemany(head + variables, data[i])

def main():
    # Cleaning data and preparing it for entry into the database.

    out = {"GageMeta": [],
           "Gage": [],
           "ReconMeta": [],
           "Recon": []}


    # Colorado River at Lees Ferry, AZ
    # Colorado River
    gage_name = "Colorado River at Lees Ferry, AZ"
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    # Now at lees_ferry_WYparsed.csv

    out["GageMeta"].append((gage_name,
                         "Upper Colorado River basin",
                         "Colorado River",
                         "m^3/s",
                         36.864722,
                         -111.5875,
                         "From personal communication with Jim Prairie",  # Citation
                         "",  # Notes
                         "http://www.usbr.gov/lc/region/g4000/NaturalFlow/current.html",  # URL
                         "2015-01-30"))
    d = pd.read_table(DATACACHE + "lees_ferry_WYparsed.csv", sep = ",",
                      skiprows = [0, 1, 2] + list(range(113, 119)),
                      header = None, names = ("year", "value"))
    d.value *= conversion_multiple
    for r in d.to_records():
        wateryear = int(r[1])
        value = float(r[2])
        out["Gage"].append((gage_name, wateryear, value))

    # Sacramento Four Rivers Index, CA
    # From personal communication with Dave Meko on 2011-11-12.
    # 	Notes on the file WSIHist_SacSan.txt are available in WSIHist_Notes.txt.
    #   A parsed version of WSIHist_SacSan.txt, including only the WY Index column,
    #   WSIHist_SacSan-PARSED.txt.
    gage_name = "Sacramento Four Rivers Index"
    conversion_multiple = 39.0875156  # MAF/yr to m^3/s

    out["GageMeta"].append((gage_name,
                         "Sacramento River basin",
                         "Sacramento River",
                         "m^3/s",
                         38.175315,
                         -121.659150,
                         "From email personal communication with Dave Meko.",  # Citation
                         "There is no reconstruction for this gage.",  # Notes
                         "",  # URL
                         "2011-11-12"))
    d = pd.read_table(DATACACHE + "WSIHist_SacSan.txt", 
                      delim_whitespace = True, 
                      skiprows = list(range(18)) + list(range(123, 131)), 
                      header = None)
    d[3] *= conversion_multiple
    for r in d.to_records():
        wateryear = int(r[1])
        value = float(r[4])
        out["Gage"].append((gage_name, wateryear, value))


    # "Jackson Lake at Dam on Snake River near Moran, WY"
    # From http://treeflow.info/pnw/index.html on 2011-11-12
    gage_name = "Jackson Lake at Dam on Snake River near Moran, WY"
    recon_name = "Reconstructed Jackson Lake at Dam on Snake River near Moran, WY"
    nan = ""
    skip_lines = 15
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 1
    recon_column = 2
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "snakemoran.csv", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Snake River Basin",
                         "Snake River",
                         "m^3/s",
                         43.858333,
                         -110.585835,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/pnw/snakemoran.html",  # URL
                         "2011-11-12"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",
                             "", # Notes
                             "http://treeflow.info/pnw/snakemoran.html",
                             "2011-11-12"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split(",")
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))


    # South Platte River at South Platte, CO
    # From http://treeflow.info/platte/splattesplatte.txt on 2012-09-04
    gage_name = "South Platte River at South Platte, CO"
    recon_name = "Reconstructed South Platte River at South Platte, CO"
    nan = "-9999"
    skip_lines = 14
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "splattesplatte.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Platte River Basin",
                         "South Platte River",
                         "m^3/s",
                         39.409167,
                         -105.169444,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/platte/splattesplatte.txt",  # URL
                         "2012-09-04"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",
                             "", # Notes
                             "http://treeflow.info/platte/splattesplatte.txt",
                             "2012-09-04"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))


    # Salt-Verde-Tonto waterways, AZ
    # From http://treeflow.info/loco/index.html on 2011-11-12
    gage_name = "Salt-Verde-Tonto waterways, AZ"
    recon_name = "Reconstructed Salt-Verde-Tonto waterways, AZ"
    nan = "-9999"
    skip_lines = 15
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "salt-verde-tonto.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Lower Colorado River basin",
                         "Salt-Verde-Tonto River",
                         "m^3/s",
                         33.537059,
                         -111.668243,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/loco/salt-verde-tonto.txt",  # URL
                         "2011-11-12"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",
                             "", # Notes
                             "http://treeflow.info/loco/salt-verde-tonto.txt",
                             "2011-11-12"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))


    # Columbia River. Recieved 2015-06-05 in personal communivation
    # from Jeremy S. Littell, USGS Research Ecologist (jlittell@usgs.gov).
    # Downloaded from http://warm.atmos.washington.edu/2860/products/sites/r7climate/subbasin_summaries/4030/nat_bias_adjusted_vic_streamflow_monthly_historical.dat on 2015-06-10.
    # NOTE: Want column 0 and a water-year sum of the other columns.
    gage_name = "Columbia River at Dalles, OR"
    out["GageMeta"].append((gage_name,
                       "Columbia River basin",
                       "Columbia River",
                       "m^3/s",
                       45.605,
                       -121.168,
                       "From email personal communication with Jeremy S. Littell.",  # Citation
                       "Downloaded 2015-06-10 from http://warm.atmos.washington.edu/2860/products/sites/r7climate/subbasin_summaries/4030/nat_bias_adjusted_vic_streamflow_monthly_historical.dat.",  # Notes
                       "",  # URL
                       "2015-06-10"))
    # This is a bit tricky because we have monthly values and need to get water-year (Oct-Sept) sums.
    d = pd.read_table(DATACACHE + "nat_bias_adjusted_vic_streamflow_monthly_historical.dat",
                      delim_whitespace = True, skiprows = 1, 
                      skip_blank_lines = True, header = None, 
                      names = ("year", "month", "value"), 
                      parse_dates = {"date": [0, 1]})
    month_list = [x.month for x in d["date"]]
    msk = np.array([x in (10, 11, 12) for x in month_list])
    wy = np.array([x.year for x in d["date"]])
    wy[msk] += 1
    d["wateryear"] = wy
    n_seconds = d.date.apply(seconds_in_month)
    d["value"] *= n_seconds
    d = d.groupby("wateryear").aggregate(np.sum)
    for r in d.to_records():
        wateryear = int(r[0])
        value = float(r[1])
        out["Gage"].append((gage_name, wateryear, value))

    ##################################### NOT CERTAIN TRINITY IS CORRECT ######
    # "Trinity River at Lewiston, CA"
    # From http://cdec.water.ca.gov/cgi-progs/queryCSV?station_id=TNL&dur_code=M&sensor_num=65&start_date=5/25/1911&end_date=Now on 2014-02-12
    gage_name = "Trinity River at Lewiston, CA"
    nan = ""
    skip_lines = 1
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 1
    conversion_multiple = 0.0283168466 # ft^3/s to m^3/s
    raw_lines = []
    with open(DATACACHE + "parsed_trinity.csv", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Trinity River Basin",
                         "Trinity River",
                         "m^3/s",
                         40.724722,
                         -122.801111,
                         "",  # Citation
                         "This has no reconstruction.",  # Notes
                         "http://cdec.water.ca.gov/cgi-progs/queryCSV?station_id=TNL&dur_code=M&sensor_num=65&start_date=5/25/1911&end_date=Now",  # URL
                         "2014-02-03"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split(",")
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), float(line[gage_column]) * conversion_multiple))


    # Rewritting database tables and inserting data.
    create_database(DB_PATH, out, SCHEMA_STRING)

if __name__ == '__main__':
    main()