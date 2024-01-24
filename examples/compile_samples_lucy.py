# For visualisation of results
import matplotlib.pyplot as plt

# For image and signal utilities
import vuba
import numpy as np
import cv2

# For exporting results into csv
import pandas as pd

import heartcv as hcv

import os
import glob
from natsort import natsorted, ns

source = '/run/media/z/rootfs/home/pi/Turner-salinity'
plots = '/run/media/z/rootfs/home/pi/Turner-salinity/cardiac_plots'

cardiac_measures = dict(position=[], timepoint=[], clutch=[], treatment=[], bpm=[], min_b2b=[], mean_b2b=[],	median_b2b=[], max_b2b=[], sd_b2b=[], range_b2b=[],	rmssd=[])
cardiac_signals = dict(position=[], timepoint=[], clutch=[], treatment=[], time=[], px_value=[], heartbeats=[])
for pos in natsorted(os.listdir(source), alg=ns.IGNORECASE):
	if '.csv' in pos or 'plots' in pos:
		continue

	print(pos)

	pos_int = int(pos[3:])
	if pos_int <= 10:
		if pos_int <= 5:
			clutch = 'A'
		else:
			clutch = 'B'
		treatment = 100
	elif pos_int > 10 and pos_int <= 20:
		if pos_int <= 15:
			clutch = 'A'
		else:
			clutch = 'B'
		treatment = 75
	elif pos_int > 20 and pos_int <= 30:
		if pos_int <= 25:
			clutch = 'A'
		else:
			clutch = 'B'
		treatment = 50
	elif pos_int > 30 and pos_int <= 40:
		if pos_int <= 35:
			clutch = 'A'
		else:
			clutch = 'B'
		treatment = 25

	tp_measures = glob.glob(f'{source}/{pos}/*_hr.csv')
	tp_signals = glob.glob(f'{source}/{pos}/*_signal.csv')

	tp_measures = natsorted(tp_measures, alg=ns.IGNORECASE)
	tp_signals = natsorted(tp_signals, alg=ns.IGNORECASE)

	for m in tp_measures:
		df = pd.read_csv(m)

		timepoint = int(str.split(str.split(m, '/')[-1], '_')[0][9:])

		cardiac_measures['position'].append(pos)
		cardiac_measures['timepoint'].append(timepoint)
		cardiac_measures['treatment'].append(treatment)
		cardiac_measures['clutch'].append(clutch)

		for k in df.keys()[1:]:
			cardiac_measures[k].append(list(df[k])[0])

	for m in tp_signals:
		df = pd.read_csv(m)

		timepoint = int(str.split(str.split(m, '/')[-1], '_')[0][9:])

		for k in range(len(df)):
			cardiac_signals['position'].append(pos)
			cardiac_signals['timepoint'].append(timepoint)
			cardiac_signals['treatment'].append(treatment)
			cardiac_signals['clutch'].append(clutch)

		for k in df.keys()[1:]:
			cardiac_signals[k].extend(list(df[k]))

		time, v, peaks = map(np.asarray, (df.time, df.px_value, df.heartbeats))
		peaks = np.where(peaks == 1)
		plt.plot(time, v, "k")
		plt.plot(time[peaks], v[peaks], "or")
		plt.xlabel("Time (seconds)")
		plt.ylabel("Mean pixel value (px)")
		plt.savefig(f'{source}/cardiac_plots/{pos}_{timepoint}.png')
		plt.clf()

cardiac_measures = pd.DataFrame(cardiac_measures)
cardiac_measures.to_csv(f'{source}/cardiac_measures.csv')

cardiac_signals = pd.DataFrame(cardiac_signals)
cardiac_signals.to_csv(f'{source}/cardiac_signals.csv')

counter = 0
for t, treat in cardiac_measures.groupby('treatment'):
	for r, rep in treat.groupby('position'):
		plt.plot(rep.timepoint, rep.bpm, 'o-', color=f'C{counter}')
	plt.plot(rep.timepoint, rep.bpm, 'o-', color=f'C{counter}', label=t)
	counter += 1

plt.xlabel('Time (hr)')
plt.ylabel('Heart rate (bpm)')
plt.legend(title='Treatment:', loc='upper right')
plt.show()












