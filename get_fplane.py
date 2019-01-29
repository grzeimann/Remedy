# -*- coding: utf-8 -*-
"""
@author: Jan Snigula
"""

try:
    # Python 3
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError:
    # Python 2
    from urllib2 import urlopen, HTTPError

import datetime
now = datetime.datetime.now()

def get_fplane(filename, datestr='', actpos=False, full=True):

    url = 'https://luna.mpe.mpg.de/fplane/' + datestr

    if actpos:
        url += '?actual_pos=1'
    else:
        url += '?actual_pos=0'

    if full:
        url += '&full_fplane=1'
    else:
        url += '&full_fplane=0'

    try:
        resp = urlopen(url)
    except HTTPError as e:
        raise Exception(' Failed to retrieve fplane file, server '
                        'responded with %d %s' % (e.getcode(), e.reason))

    with open(filename, 'w') as f:
        f.write(resp.read().decode())

date = '%04d%02d%02d' % (now.year, now.month, now.day)
get_fplane('fplane%s.txt' % date)