To create a new calibration file:
1) Log on to TACC
2) Get Remedy if you haven't already on TACC
3) Add the following to your ~/.bashrc file if you haven't already
	export PATH=/home/00115/gebhardt/anaconda2/bin:/work/03946/hetdex/maverick/bin:$PATH
4) To create a new calibration file, run:
	python Remedy/create_cal_hdf5.py -d 20190112 -o 27 -of test_cal_20190112.h5 -r "/work/03946/hetdex/maverick/red1/reductions"

	In this case I choose the night 20190112, observation 27, and called it test_cal_20190112.h5
	The night and observation were ones of quality reduction, most importantly with respect to wavelength and trace, which is all that is used
5) Copy calibration file to your desired Remedy/CALS folder
	cd WHEREVER/Remedy/CALS
	scp username@wrangler.tacc.utexas.edu:/PATHONTACC/test_cal_20190112.h5 .
