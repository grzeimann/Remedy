# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:28:59 2017

@author: gregz
"""

try:
    import httplib
except:
    import httplib2 as httplib
import json
import sys
import warnings
import requests
import uuid
import os

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse_single_table
from astropy.table import Table
from astropy.vo.client import vos_catalog
from urllib import pathname2url as urlencode


with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from astroquery.sdss import SDSS


class MakeRegionFile(object):
    @classmethod
    def writeHeader(cls, f):
        """Write the header to file ``f``

        Parameters
        ----------
        f : file-like object
            where to write to; must have a ``write`` method
        """
        s = []

        s.append('# Region file format: DS9 version 4.1')
        s.append('global color=green dashlist=8 3 width=1 '
                 'font="helvetica 10 normal roman" select=1 highlite=1 dash=0 '
                 'fixed=0 edit=1 move=1 delete=1 include=1 source=1')
        s.append('fk5')
        f.write('\n'.join(s) + "\n")

    @classmethod
    def writeSource(cls, f, ra, dec, rad=2):
        s = []
        for r, d in zip(ra, dec):
            s.append('circle(%0.6f, %0.6f, %0.2f")' % (r, d, rad))
        f.write('\n'.join(s) + '\n')
        f.flush()


def mastQuery(request):
    ''' Mast seems to respond quickly which is great '''
    server = 'mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/" + version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')
    # Close the https connection
    conn.close()

    return head, content


def mastJson2Table(jsonObj):
    ''' Table() allows for numpy array's and easy calls '''
    dataTable = Table()
    for col, atype in [(x['name'], x['type']) for x in jsonObj['fields']]:
        if atype == 'string':
            atype = 'str'
        if atype == 'boolean':
            atype = 'bool'
        tcol = []
        for x in jsonObj['data']:
            col_val = x.get(col, None)
            if col_val is None:
                tcol.append(-999.)
            else:
                tcol.append(col_val)
        dataTable[col] = np.array(tcol, dtype=atype)
    return dataTable


def queryTESS_IC(ra, dec, radius):
    ''' TESS input catalog is a mash-up of 2mass, SDSS, GAIA, and more '''
    mashupRequest = {'service': 'Mast.Catalogs.Tic.Cone',
                     'params': {'ra': ra,
                                'dec': dec,
                                'radius': radius},
                     'format': 'json',
                     'pagesize': 10000,
                     'page': 1}

    headers, outString = mastQuery(mashupRequest)

    outData = json.loads(outString)

    table = mastJson2Table(outData)
    sel = np.where(table['objType'] == 'STAR')[0]
    return table[sel]


def panstarrs_query(ra_deg, dec_deg, rad_deg, mindet=1,
                    maxsources=30000,
                    server=('https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.votable')):
    """
    Query Pan-STARRS DR1 @ MAST
    parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field
                                          radius in degrees
                mindet: minimum number of detection (optional)
                maxsources: maximum number of sources
                server: servername
    returns: astropy.table object
    """
    r = requests.get(server, params={'ra': ra_deg, 'dec': dec_deg,
                                     'radius': rad_deg, 'pagesize': maxsources,
                                     'nDetections.gte': ('%d' % mindet)})

    # write query data into local file
    name = str(uuid.uuid4()) + '.xml'
    outf = open(name, 'w')
    outf.write(r.text)
    outf.close()
    # parse local file into astropy.table object
    data = parse_single_table(name)
    os.remove(name)
    return data.to_table(use_names_over_ids=True)


def query_panstarrs(ra, dec, radius, debug=False):
    """
    Queries PANSTARRS
    """
    T = panstarrs_query(ra, dec, radius)
    sel1 = (~T['gMeanPSFMag'].mask) * (~T['rMeanPSFMag'].mask) * (~T['iMeanPSFMag'].mask)
    T = T[sel1]
    return T


def queryUSNO_A2(ra, dec, radius):
    '''
    Queries USNO_A2.
    ra  = center RA of field
    dec = center DEC of field
    radius = determines size of radius around ra/dec for which sources should
              be retrieved
    return array of stars with format
    IDa IDb RA DEC 0. B R 0. 0.
    '''

    usno_a2_name = 'The USNO-A2.0 Catalogue (Monet+ 1998) 1'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vos_catalog.call_vo_service('conesearch_good', verbose=False,
                                             kwargs={'RA': ra, 'DEC': dec,
                                                     'SR': radius, 'VERB': 1,
                                                     'cftype': 'ASCII'},
                                             catalog_db=usno_a2_name)
    table = Table(result.array.data)
    table['RAJ2000'].name = 'ra'
    table['DEJ2000'].name = 'dec'
    B, V = (table['Bmag'], table['Rmag'])
    table['gmag'] = V + 0.06 * (B - V) - 0.12  # rough conversion

    return table


def querySDSS(ra, dec, radius):
    ''' using astroquery sdss system '''
    pos = SkyCoord(ra*u.deg, dec*u.deg, frame='fk5')
    table = SDSS.query_region(pos, radius=radius*u.deg,
                              photoobj_fields=['ra', 'dec', 'objid', 'type',
                                               'u', 'g', 'r', 'i', 'z'])
    return table[np.where(table['type'] == 6)[0]]
