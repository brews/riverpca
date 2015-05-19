# IPython log file

get_ipython().magic('cd /home/sbm/Desktop/river modes')
import pygrib
get_ipython().magic('logon')
get_ipython().magic('logstart ')
grbs = pygrib.open("ei.moda.an.pl.regn128sc.1979010100")
grb = grbs.select(shortName = "gh", level = 500, typeOfLevel = "isobaricInhPa")[0]
dir(grbs)
grbs.readline()
grbs.readline()
grb = grbs.select(shortName = "gh", level = 500)
grb = grbs.select(shortName = "gh")
grb = grbs.select(shortName = "avgua")
grbs.tell()
grbs.read(1)[0]
grbs.seek(0)
for grb in grbs:
    grb
    
for grb in grbs:
    print(grb)
    
grbs = pygrib.open("ei.moda.an.pl.regn128sc.1979010100")
for grb in grbs:
    grb
    
grbs.seek(0)
for grb in grbs:
    grb
    
grbs.message()
grbs.message(1)
grbs.message(0)
grbs.message(2)
grb = grbs.select(name = "Geopotential", level = 500)
grb[0]
grb = grbs.select(name = "Geopotential", level = 500, typeOfLevel = "isobaricInhPa")
exit()
