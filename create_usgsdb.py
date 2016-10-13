#! /usr/bin/env python3
# 2016-03-06

#HCDN-2009 site http://water.usgs.gov/osw/hcdn-2009/

# Example query:
#http://waterservices.usgs.gov/nwis/stat/?format=rdb&sites=01646500&parameterCd=00060&statReportType=annual&statYearType=water

# Create an SQLite3 database containing HCDN-2009 data.

import urllib.request
import sqlite3 as sql

WY_LINE_SKIP = 33
MONTHLY_LINE_SKIP = 34
DB_PATH = "./data/stationdb.sqlite"
SCHEMA_STRING = """
            DROP TABLE IF EXISTS StationInfo;
            CREATE TABLE StationInfo(id INTEGER PRIMARY KEY,
                                   stationid TEXT,
                                   stationname TEXT,
                                   class TEXT,
                                   aggecoregion TEXT,
                                   drainsqkm REAL,
                                   huc02 TEXT,
                                   latgage REAL,
                                   longage REAL,
                                   state TEXT,
                                   active09 TEXT,
                                   flowyrs19002009 INTEGER,
                                   flowyrs19502009 INTEGER,
                                   flowyrs19902009 INTEGER);
            DROP TABLE IF EXISTS StationWY;
            CREATE TABLE StationWY(id INTEGER PRIMARY KEY,
                               agency TEXT,
                               stationid TEXT,
                               parameter TEXT,
                               dd INTEGER,
                               locweb TEXT,
                               year INTEGER,
                               meanCFS REAL,
                               count INTEGER,
                               FOREIGN KEY (stationid) REFERENCES StationInfo(stationid));
            DROP TABLE IF EXISTS StationMonthly;
            CREATE TABLE StationMonthly(id INTEGER PRIMARY KEY,
                               agency TEXT,
                               stationid TEXT,
                               parameter TEXT,
                               dd INTEGER,
                               locweb TEXT,
                               year INTEGER,
                               month INTEGER,
                               meanCFS REAL,
                               count INTEGER,
                               wy INTEGER,
                               FOREIGN KEY (stationid) REFERENCES StationInfo(stationid));
                """

def create_database(path, data, schema):
    """Create an sqlite3 database of river gage information.

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

def get_monthly(x, skip_lines=0):
    """Get monthly USGS gage dischage from station ID `x`."""
    url_target = "http://waterservices.usgs.gov/nwis/stat/?format=rdb&sites=%s&parameterCd=00060&statReportType=monthly" % x
    response = urllib.request.urlopen(url_target)
    raw_lines = response.readlines()
    response.close()
    for line in raw_lines[skip_lines:]:
        if (line[0] == b"#") or (line == b"agency_cd\tsite_no\tparameter_cd\tdd_nu\tloc_web_ds\tyear_nu\tmonth_nu\tmean_va\tcount_nu\n") or (line == b"5s\t15s\t5s\t3n\t15s\t4s\t2s\t12n\t8n\n"):
          continue
        line = line.decode("utf-8")
        line = line.rstrip().strip().split("\t")
        wy = int(line[5])
        if int(line[6]) >= 10:
          wy += 1
        line.append(str(wy))
        yield(tuple(line))

def get_wateryear(x, skip_lines=0):
    """Get water year USGS gage dischage from station ID `x`."""
    url_target = "http://waterservices.usgs.gov/nwis/stat/?format=rdb&sites=%s&parameterCd=00060&statReportType=annual&statYearType=water" % x
    response = urllib.request.urlopen(url_target)
    raw_lines = response.readlines()
    response.close()
    for line in raw_lines[skip_lines:]:
        line = line.decode("utf-8")
        line = line.rstrip().strip().split("\t")
        yield(tuple(line))


def main():
    # Cleaning data and preparing it for entry into the database.

    out = {"StationInfo": [], "StationWY": [], "StationMonthly": []}

    # Read & process HCDN-2009 file.
    skip_lines = 1
    with open("./data/HCDN-2009_Station_Info.tsv", "r") as fl:
        raw_lines = fl.readlines()
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().strip().split("\t")
        # For whatever reason some of the station IDs are missing a "0".
        if len(line[0]) == 7:
            line[0] = "0" + line[0]
        wy_file = get_wateryear(line[0], WY_LINE_SKIP)
        for l in wy_file:
            out["StationWY"].append(l)
        monthly_file = get_monthly(line[0], MONTHLY_LINE_SKIP)
        for l in monthly_file:
            out["StationMonthly"].append(l)
        out["StationInfo"].append(tuple(line))

    # Rewritting database tables and inserting data.
    create_database(DB_PATH, out, SCHEMA_STRING)

if __name__ == '__main__':
    main()
