#! /usr/bin/env python3
# Create an SQLite3 database containing river gage data, reconstructions
# and metadata. We're also carefully cleaning the data here.

import sqlite3 as sql

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

    # Arkansas River at Canon City, CO
    # From http://treeflow.info/ark/index.html on 2011-11-12
    # Arkansas River
    gage_name = "Arkansas River at Canon City, CO"
    recon_name = "Reconstructed Arkansas River at Canon City, CO"
    nan = "-9999"
    skip_lines = 14
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "arkansascanoncity.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Arkansas River basin",
                         "Arkansas River",
                         "m^3/s",
                         38.433889,
                         -105.256667,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/ark/arkansascanoncity.txt",  # URL
                         "2011-11-12"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",  # Citation
                             "", # Notes
                             "http://treeflow.info/ark/arkansascanoncity.txt",
                             "2011-11-12"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))

    # Colorado River at Lees Ferry, AZ
    # From http://treeflow.info/upco/index.html on 2011-11-12
    # Colorado River
    gage_name = "Colorado River at Lees Ferry, AZ"
    recon_name = "Reconstructed Colorado River at Lees Ferry, AZ"
    nan = "-9999"
    skip_lines = 17
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "coloradoleesmeko.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Upper Colorado River basin",
                         "Colorado River",
                         "m^3/s",
                         36.864722,
                         -111.5875,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/upco/coloradoleesmeko.txt",  # URL
                         "2011-11-12"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "Meko, D.M., C.A. Woodhouse, C.H. Baisan, T. Knight, J.J. Lukas, M.K. Hughes, and M.W. Salzer. 2007. Medieval drought in the upper Colorado River Basin. Geophysical Research Letters, Vol. 34, L10705",
                             "", # Notes
                             "http://treeflow.info/upco/coloradoleesmeko.txt",
                             "2011-11-12"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))

    # Missouri River. Recieved 2011-11-18 in personal communivation
    # from Connie Woodhouse.
    # NOTE: Want column 0 and a water-year sum of the other columns.
    gage_name = "Missouri River at Toston, MT"
    nan = None
    skip_lines = 6
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = None
    recon_column = None
    conversion_multiple = 0.0390875156 # thousand acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "Natural_Flows_Missouri_Toston.csv", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Upper Missouri River basin",
                         "Missouri River",
                         "m^3/s",
                         46.146111,
                         -111.419722,
                         "From email personal communication with Connie Woodhouse.",  # Citation
                         "Recieved 2011-11-18 in personal communivation from Connie Woodhouse.",  # Notes
                         "",  # URL
                         "2011-11-18"))
    # This is a bit tricky because we have monthly values and need to get water-year (Oct-Sept) sums.
    prev_cal_year = None
    prev_cal_year_sum = None
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split(",")
        cal_year = int(line[0])
        if prev_cal_year == (cal_year - 1):
            water_year_value = sum([float(i) for i in line[1:10]]) + prev_cal_year_sum
            out["Gage"].append((gage_name, cal_year, water_year_value * conversion_multiple))
        prev_cal_year_sum = sum([float(i) for i in line[10:13]])
        prev_cal_year = cal_year

    # Rio Grande at Otowi Bridge, NM
    # From http://treeflow.info/riogr/riograndeotowinrcs.txt on 2012-09-04
    gage_name = "Rio Grande at Otowi Bridge, NM (NRCS flows)"
    recon_name = "Reconstructed Rio Grande at Otowi Bridge, NM (NRCS flows)"
    nan = "-9999"
    skip_lines = 14
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "riograndeotowinrcs.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Rio Grande River basin",
                         "Rio Grande River",
                         "m^3/s",
                         35.8745,
                         -106.142444,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/riogr/riograndeotowinrcs.txt",  # URL
                         "2012-09-04"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",
                             "", # Notes
                             "http://treeflow.info/riogr/riograndeotowinrcs.txt",
                             "2012-09-04"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))

    # Sacramento Four Rivers Index, CA
    # From personal communication with Dave Meko on 2011-11-12.
    # 	Notes on the file WSIHist_SacSan.txt are available in WSIHist_Notes.txt.
    #   A parsed version of WSIHist_SacSan.txt, including only the WY Index column,
    #   WSIHist_SacSan-PARSED.txt.
    gage_name = "Sacramento Four Rivers Index"
    nan = ""
    skip_lines = 1
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 1
    recon_column = None
    conversion_multiple = 39.0875156  # MAF/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "WSIHist_SacSan-PARSED.txt", "r") as fl:
        raw_lines = fl.readlines() 

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
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), float(line[gage_column]) * conversion_multiple))

    # Salinas River at Paso Robles, CA
    # From http://treeflow.info/cali/salinas.txt on 2011-11-12
    gage_name = "Salinas River at Paso Robles, CA"
    recon_name = "Reconstructed Salinas River at Paso Robles, CA"
    nan = "-9999"
    skip_lines = 18
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 0.0283168466 # ft^3/s to m^3/s
    raw_lines = []
    with open(DATACACHE + "salinas.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Salinas River Basin",
                         "Salinas River",
                         "m^3/s",
                         35.628611,
                         -120.683333,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/cali/salinas.txt",  # URL
                         "2011-11-12"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",
                             "", # Notes
                             "http://treeflow.info/cali/salinas.txt",
                             "2011-11-12"))
    for line in raw_lines[skip_lines:]:
      line = line.rstrip().split()
      if line[gage_column] != nan:
          out["Gage"].append((gage_name, int(line[year_column]), float(line[gage_column]) * conversion_multiple))
      if line[recon_column] != nan:
          out["Recon"].append((gage_name, int(line[year_column]), float(line[recon_column]) * conversion_multiple))

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

    # Green River near Green River, WY
    # from http://treeflow.info/upco/greengreenrivwybarnett.txt on 2013-06-27
    gage_name = "Green River near Green River, WY"
    recon_name = "Reconstructed Green River near Green River, WY"
    nan = "-9999"
    skip_lines = 16
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "greengreenrivwybarnett.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Upper Colorado River basin",
                         "Green River",
                         "m^3/s",
                         41.516389,
                         -109.448333,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/upco/greengreenrivwybarnett.html",  # URL
                         "2014-01-31"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",
                             "", # Notes
                             "http://treeflow.info/upco/greengreenrivwybarnett.html",
                             "2014-01-31"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))

    # Green River near Greendale, UT
    # from http://treeflow.info/upco/greengreendale.html
    gage_name = "Green River near Greendale, UT"
    recon_name = "Reconstructed Green River near Greendale, UT"
    nan = "-9999"
    skip_lines = 16
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 3.90875156e-5 # acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "greengreendale.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Upper Colorado River basin",
                         "Green River",
                         "m^3/s",
                         40.908333,
                         -109.422222,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/upco/greengreendale.html",  # URL
                         "2014-01-31"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",
                             "", # Notes
                             "http://treeflow.info/upco/greengreendale.html",
                             "2014-01-31"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))

    # Cache la Poudre River at Canyon Mouth, CO
    # from http://treeflow.info/platte/poudre.html
    gage_name = "Cache la Poudre River at Canyon Mouth, CO"
    recon_name = "Reconstructed Cache la Poudre River at Canyon Mouth, CO"
    nan = "-9999"
    skip_lines = 16
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 2
    recon_column = 1
    conversion_multiple = 0.0390875156 # 1000 acre feet/yr to m^3/s
    raw_lines = []
    with open(DATACACHE + "poudre.txt", "r") as fl:
        raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                         "Platte River basin",
                         "Cache la Poudre River",
                         "m^3/s",
                         40.664444,
                         -105.223889,
                         "",  # Citation
                         "",  # Notes
                         "http://treeflow.info/platte/poudre.html",  # URL
                         "2014-01-31"))
    out["ReconMeta"].append((recon_name,
                             "m^3/s",
                             gage_name,
                             "",
                             "", # Notes
                             "http://treeflow.info/platte/poudre.html",
                             "2014-01-31"))
    for line in raw_lines[skip_lines:]:
        line = line.rstrip().split()
        if line[gage_column] != nan:
            out["Gage"].append((gage_name, int(line[year_column]), int(line[gage_column]) * conversion_multiple))
        if line[recon_column] != nan:
            out["Recon"].append((gage_name, int(line[year_column]), int(line[recon_column]) * conversion_multiple))
    
    # Columbia River at Dalles. Recieved 2011-11-18 in personal communivation
    # from Connie Woodhouse.
    # TODO: Check the units on this. Assuming it's in ft^3/s and in wateryears?
    gage_name = "Columbia River at Dalles, OR"
    recon_name = None
    nan = None
    skip_lines = 1
    # The values for each of the columns in the raw file, starting from 0.
    year_column = 0
    gage_column = 1
    recon_column = None
    conversion_multiple = 0.0283168466 # ft^3/s to m^3/s
    raw_lines = []
    with open(DATACACHE + "ColumbiaFlow_Hamlet.csv", "r") as fl:
      raw_lines = fl.readlines() 

    out["GageMeta"].append((gage_name,
                       "Columbia River basin",
                       "Columbia River",
                       "m^3/s",
                       45.6075,
                       -121.172222,
                       "From email personal communication with Connie Woodhouse.",  # Citation
                       "Recieved 2011-11-18 in personal communivation from Connie Woodhouse.",  # Notes
                       "",  # URL
                       "2011-11-18"))
    for line in raw_lines[skip_lines:]:
      line = line.rstrip().split(",")
      out["Gage"].append((gage_name, int(line[year_column]), float(line[gage_column]) * conversion_multiple))

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