"""Parse HCDN-2009 data from USGS online waterservice
"""

import logging
import sqlite3 as sql
import pandas as pd

# If this Excel sheet disappears from the USGS site below, I left a 
# tab-delimited copy in the ../data/ directory.
META_PATH = 'http://water.usgs.gov/osw/hcdn-2009/HCDN-2009_Station_Info.xlsx'

def _read_gage(site, reporttype_str, skip_rows):
    """Get USGS waterservice information
    """
    base_url = 'http://waterservices.usgs.gov/nwis/stat/'
    query_template = '?format=rdb&sites={sitecode}&parameterCd=00060&{reporttype}'
    query_str = query_template.format(sitecode = site,
                                      reporttype = reporttype_str)
    url_target = base_url + query_str
    d = pd.read_table(url_target, skiprows = skip_rows, dtype = {'site_no': 'str'})
    return d

def load_wy(site):
    """Load wateryear gage reports from USGS waterservice
    """
    report_type = 'statReportType=annual&statYearType=water'
    srows = list(range(31)) + [32]
    d = _read_gage(site, report_type, srows)
    return d

def load_month(site):
    """Load monthly gage reports from USGS waterservice
    """
    report_type = 'statReportType=monthly'
    srows = list(range(32)) + [33]
    d = _read_gage(site, report_type, srows)
    return d

def clean_sitecode(x):
    """Parser for HCDN-2009 metadata station IDs

    The site ID's used by USGS are often ID'd by Excel and other programs as 
    integers and not strings. This is a problem because leading '0's are 
    often cut off the ID in the conversion and then USGS web services cannot 
    match when we query them. So, we're converting the IDs to character 
    strings and stuffing the short IDs with '0's. 
    """
    x = str(x)
    if len(x) == 7:
        x = '0' + x
    return x

def load_metadata(path=META_PATH):
    """Read in metadata Excel spreadsheet for USGS HCDN-2009 gages
    """
    meta = pd.read_excel(path, converters = {'STATION ID': clean_sitecode})
    return meta

def hcdn(outpath):
    """Read USGS HCDN meta data and load gages into sqlite DB
    """
    logging.debug('Loading USGS HCDN metadata')
    meta = load_metadata()
    all_wy = pd.DataFrame()
    all_month = pd.DataFrame()
    for x, row in meta.iterrows():
        site = row['STATION ID']
        logging.debug('Processing site ' + site)
        try:
            all_wy = all_wy.append(load_wy(site), ignore_index = True)
            all_month = all_month.append(load_month(site), ignore_index = True)
        except pd.io.common.EmptyDataError:
            # This error is thrown when USGS waterservice can't find the gage.
            logging.warning('Skipping site ' + site + ' as found EmptyDataError found')
            continue
    with sql.connect(outpath) as con:
        all_wy.to_sql('StationWY', con)
        all_month.to_sql('StationMonthly', con)
        meta.to_sql('StationInfo', con)

