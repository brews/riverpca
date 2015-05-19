#! /usr/bin/env python
# 2015-01-06
# Download ERA-I data from rda.ucar.edu.
#
# python script to download selected files from rda.ucar.edu
# after you save the file, don't forget to make it executable
#   i.e. - "chmod 755 <name_of_script>"
#
import sys
import os
import urllib2
import cookielib
#
if (len(sys.argv) != 2):
  print "usage: "+sys.argv[0]+" [-q] password_on_RDA_webserver"
  print "-q suppresses the progress message for each file that is downloaded"
  sys.exit(1)
#
passwd_idx=1
verbose=True
if (len(sys.argv) == 3 and sys.argv[1] == "-q"):
  passwd_idx=2
  verbose=False
#
cj=cookielib.MozillaCookieJar()
opener=urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
#
# check for existing cookies file and authenticate if necessary
do_authentication=False
if (os.path.isfile("auth.rda.ucar.edu")):
  cj.load("auth.rda.ucar.edu",False,True)
  for cookie in cj:
    if (cookie.name == "sess" and cookie.is_expired()):
      do_authentication=True
else:
  do_authentication=True
if (do_authentication):
  login=opener.open("https://rda.ucar.edu/cgi-bin/login","email=malevich@email.arizona.edu&password="+sys.argv[1]+"&action=login")
#
# save the authentication cookies for future downloads
# NOTE! - cookies are saved for future sessions because overly-frequent authentication to our server can cause your data access to be blocked
  cj.clear_session_cookies()
  cj.save("auth.rda.ucar.edu",True,True)
#
# download the data file(s)
listoffiles=["ei.moda.an.pl/ei.moda.an.pl.regn128sc.1979010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1979020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1979030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1979040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1979110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1979120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1980010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1980020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1980030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1980040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1980110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1980120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1981010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1981020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1981030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1981040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1981110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1981120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1982010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1982020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1982030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1982040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1982110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1982120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1983010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1983020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1983030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1983040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1983110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1983120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1984010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1984020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1984030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1984040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1984110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1984120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1985010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1985020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1985030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1985040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1985110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1985120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1986010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1986020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1986030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1986040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1986110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1986120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1987010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1987020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1987030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1987040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1987110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1987120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1988010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1988020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1988030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1988040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1988110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1988120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1989010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1989020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1989030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1989040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1989110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1989120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1990010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1990020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1990030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1990040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1990110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1990120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1991010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1991020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1991030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1991040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1991110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1991120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1992010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1992020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1992030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1992040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1992110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1992120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1993010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1993020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1993030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1993040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1993110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1993120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1994010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1994020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1994030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1994040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1994110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1994120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1995010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1995020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1995030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1995040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1995110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1995120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1996010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1996020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1996030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1996040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1996110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1996120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1997010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1997020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1997030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1997040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1997110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1997120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1998010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1998020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1998030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1998040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1998110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1998120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1999010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1999020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1999030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1999040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1999110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.1999120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2000010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2000020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2000030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2000040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2000110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2000120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2001010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2001020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2001030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2001040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2001110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2001120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2002010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2002020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2002030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2002040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2002110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2002120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2003010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2003020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2003030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2003040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2003110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2003120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2004010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2004020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2004030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2004040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2004110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2004120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2005010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2005020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2005030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2005040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2005110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2005120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2006010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2006020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2006030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2006040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2006110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2006120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2007010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2007020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2007030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2007040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2007110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2007120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2008010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2008020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2008030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2008040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2008110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2008120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2009010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2009020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2009030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2009040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2009110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2009120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2010010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2010020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2010030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2010040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2010110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2010120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2011010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2011020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2011030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2011040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2011110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2011120100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2012010100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2012020100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2012030100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2012040100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2012110100",
"ei.moda.an.pl/ei.moda.an.pl.regn128sc.2012120100"]
for file in listoffiles:
  idx=file.rfind("/")
  if (idx > 0):
    ofile=file[idx+1:]
  else:
    ofile=file
  if (verbose):
    sys.stdout.write("downloading "+ofile+"...")
    sys.stdout.flush()
  infile=opener.open("http://rda.ucar.edu/data/ds627.1/"+file)
  outfile=open(ofile,"wb")
  outfile.write(infile.read())
  outfile.close()
  if (verbose):
    sys.stdout.write("done.\n")