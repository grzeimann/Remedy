# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:52:35 2017

@author: gregz
"""
import glob

import tables as tb
import argparse as ap
import os.path as op

from astropy.io import fits
from input_utils import setup_logging


def build_path(reduction_folder, instr, date, obsid, expn):
    folder = op.join(date, instr, "{:s}{:07d}".format(instr, int(obsid)),
                     "exp{:02d}".format(int(expn)), instr)
    return op.join(reduction_folder, folder)


def get_files(args):
    files = glob.glob(op.join(args.rootdir, args.date, 'virus',
                              'virus%07d' % int(args.observation),
                              'exp01', 'virus', 'multi_*.fits'))
    return files


class VIRUSImage(tb.IsDescription):
    wavelength = tb.Float32Col((112, 1032))
    trace = tb.Float32Col((112, 1032))
    ifupos = tb.Float32Col((112, 2))
    ifuslot = tb.Int32Col()
    ifuid = tb.StringCol(3)
    specid = tb.StringCol(3)
    amp = tb.StringCol(2)


def append_fibers_to_table(im, fn):
    F = fits.open(fn)
    imattr = ['wavelength', 'trace', 'ifupos']
    for att in imattr:
        if att == 'image':
            im[att] = F['PRIMARY'].data * 1.
        else:
            im[att] = F[att].data * 1.
    im['ifuslot'] = int(F[0].header['IFUSLOT'])
    im['ifuid'] = '%03d' % int(F[0].header['IFUID'])
    im['specid'] = '%03d' % int(F[0].header['SPECID'])
    im['amp'] = '%s' % F[0].header['amp'][:2]
    im.append()
    return True


def main(argv=None):
    ''' Main Function '''
    # Call initial parser from init_utils
    parser = ap.ArgumentParser(description="""Create HDF5 file.""",
                               add_help=True)

    parser.add_argument("-d", "--date",
                        help='''Date, e.g., 20170321, YYYYMMDD''',
                        type=str, default=None)

    parser.add_argument("-o", "--observation",
                        help='''Observation number, "00000007" or "7"''',
                        type=str, default=None)

    parser.add_argument("-r", "--rootdir",
                        help='''Root Directory for Reductions''',
                        type=str, default='/work/03946/hetdex/maverick')

    parser.add_argument('-of', '--outfilename', type=str,
                        help='''Relative or absolute path for output HDF5
                        file.''', default=None)

    parser.add_argument('-a', '--append',
                        help='''Appending to existing file.''',
                        action="count", default=0)

    args = parser.parse_args(argv)
    args.log = setup_logging()

    # Get the daterange over which reduced files will be collected
    files = get_files(args)

    # Creates a new file if the "--append" option is not set or the file
    # does not already exist.
    does_exist = False
    if op.exists(args.outfilename) and args.append:
        fileh = tb.open_file(args.outfilename, 'a')
        does_exist = True
    else:
        fileh = tb.open_file(args.outfilename, 'w')
        group = fileh.create_group(fileh.root, 'Info',
                                   'Cal Data and Metadata')
        imagetable = fileh.create_table(group, 'Cals', VIRUSImage, 'Cal Info')

    for fn in files:
        args.log.info('Working on %s' % fn)
        im = imagetable.row
        success = append_fibers_to_table(im, fn)
        if success:
            imagetable.flush()
    
    fileh.close()


if __name__ == '__main__':
    main()
