import matplotlib.pyplot as plt
import numpy as np
import heartcv as hcv
import pandas as pd
import glob
from statsmodels.nonparametric.smoothers_lowess import lowess
from more_itertools import pairwise

from heartcv import minmax_scale as scale

def mse(truth, act):
    truth, act = map(np.asarray, (truth, act))
    truth, act = map(scale, (truth, act))
    diff = truth - act
    return np.nanmean(diff**2)

def rmse(truth, act):
    return np.sqrt(mse(truth, act))

def between(arr, from_, to_):
    at = []
    for f,t in zip(from_, to_):
        inds = list(range(f, t+1))
        ret = arr[slice(f, t+1, None)]
        at.append((inds, ret))
    return at

def systole(arr, peaks, troughs):
    min_p, min_t = map(np.min, (peaks, troughs))
    if min_p < min_t:
        peaks = peaks[1:]

    return between(arr, troughs, peaks)

def diastole(arr, peaks, troughs):
    min_p, min_t = map(np.min, (peaks, troughs))
    if min_p > min_t:
        troughs = troughs[1:]

    return between(arr, peaks, troughs)

def parse_auto(file, inv, subset, rem_peak=None, idx=1, plot=False, *args, **kwargs):
    dat = pd.read_csv(file)
    area = dat['area']
    if inv:
        area = np.max(area) - area
    area = scale(area)[:300] # to 300 as manual is only up to 300
    t,d,s = hcv.find_events(area, *args, **kwargs)
    d = d[0]
    s = s[0]
    d,s = map(np.asarray, (d,s))

    print(d)
    if rem_peak:
        d = d[d != rem_peak]
    d = np.asarray([_d for i,_d in enumerate(d) if i%2 == idx])

    sys = []
    for _d in d:
        diff_s = s - _d
        pos_diff_s = diff_s[diff_s > 0]
        if len(pos_diff_s):
            _s = np.min(pos_diff_s)
            sys.append(_s + _d)
        else:
            idx = d.tolist().index(_d)
            d = np.delete(d, idx)

    s = np.asarray(sys)

    if subset:
        d_to = subset.pop('d', None)
        s_to = subset.pop('s', None)

        if d_to:
            d = d[:d_to]
        if s_to:
            s = s[s_to:]

    if plot:
        plt.plot(area)
        plt.plot(d, area[d], 'x')
        plt.plot(s, area[s], 'x')
        plt.show()

    print(tuple(map(len, (d,s))))

    return (d,area[d],s,area[s])

def parse_man(file, subset=None):
    dat = pd.read_csv(file)
    d_peaks = dat['EndDiastoleFrame']
    s_peaks = dat['EndSystoleFrame']

    d_areas = dat['EndDiastoleArea']
    s_areas = dat['EndSystoleArea']

    if subset:
        d_to = subset.pop('d', None)
        s_to = subset.pop('s', None)

        if d_to:
            d_peaks = d_peaks[d_to]
        if s_to:
            s_peaks = s_peaks[s_to]

    return tuple(map(np.asarray, (d_peaks, s_peaks))), (d_areas, s_areas)

def stats(d_peaks, s_peaks):
    # HR - bpm
    # Diastole timing
    # Systole timing
    # Rate of diastole
    # Rate of systole

    # Beat to beat variability
    # i) Minimum timing difference between peaks
    # ii) Maximum timing difference between peaks
    # iii) Mean timing difference between peaks 
    # iv) Median timing difference between peaks
    # v) Range of timing difference between peaks
    # vi) Standard deviation in timing difference between peaks.
    # vii) Root mean square of successive difference in the timing between peaks

    hr = len(s_peaks)

    d_time = (s_peaks[:-1] - d_peaks[1:])*-1
    s_time = s_peaks - d_peaks

    # Times
    min_dt = d_time.min()
    max_dt = d_time.max()
    mean_dt = d_time.mean()
    med_dt = np.median(d_time)
    std_dt = d_time.std()

    min_st = s_time.min()
    max_st = s_time.max()
    mean_st = s_time.mean()
    med_st = np.median(s_time)
    std_st = s_time.std()

    # Diffs
    d_diffs = d_peaks[1:] - d_peaks[:-1] # only use diastole as more reliablly accurate
    min_t = d_diffs.min()
    max_t = d_diffs.max()
    mean_t = d_diffs.mean()
    med_t = np.median(d_diffs)
    std_t = d_diffs.std()
    
    # RMSS
    rmss = rmse(d_diffs[1:], d_diffs[:-1])    

    return [[hr], 
            [min_t], [max_t], [mean_t], [med_t], [std_t], [rmss], 
            [min_dt], [max_dt], [mean_dt], [med_dt], [std_dt], 
            [min_st], [max_st], [mean_st], [med_st], [std_st]]

def append_with(to, with_):
    new = []
    for lt,lw in zip(to, with_):
        new.append(lt + lw)
    return new

# Auto
# ds = parse_auto('./data/paleomon/fd_auto_15_15ppt_medium_1.csv', inv=False, subset=None, idx=0, prominence=0.3, width=(1,3), distance=4) 
# ds1 = parse_auto('./data/paleomon/fd_auto_15_15ppt_old_1.csv', inv=False, subset=None, idx=0, prominence=0.1) # s[1:]
ds2 = parse_auto('./data/paleomon/fd_auto_15_15ppt_young_1.csv', inv=False, subset=None, idx=0, prominence=0.1) # d[:-1], s[1:]
# ds3 = parse_auto('./data/paleomon/fd_auto_15_15ppt_young_A1_37.csv', inv=False, subset=None, idx=0, prominence=0.1)
ds4 = parse_auto('./data/paleomon/fd_auto_15_15ppt_young_A1_60.csv', inv=False, subset=None, idx=0, prominence=0.1)
ds5 = parse_auto('./data/paleomon/fd_auto_15_15ppt_young_A4_60.csv', inv=False, subset=None, idx=0, prominence=0.1) # s[1:]
# ds6 = parse_auto('./data/paleomon/fd_auto_15_15ppt_young_C12_1.csv', inv=False, subset=None, idx=0, prominence=0.1) # s[1:]
# ds7 = parse_auto('./data/paleomon/fd_auto_15_15ppt_young_D3_50.csv', inv=False, subset=None, idx=0, prominence=0.1) # s[1:]
# ds8 = parse_auto('./data/paleomon/fd_auto_15_15ppt_young_H3_90.csv', inv=False, subset=None, idx=0, prominence=0.1) # s[1:]
ds9 = parse_auto('./data/paleomon/fd_auto_15_15ppt_medium_B2_60.csv', inv=False, subset=None, idx=1, prominence=0.1) 
ds10 = parse_auto('./data/paleomon/fd_auto_15_15ppt_medium_C2_149.csv', inv=False, subset=None, idx=0, prominence=0.1) 
# ds11 = parse_auto('./data/paleomon/fd_auto_15_15ppt_medium_C5_1.csv', inv=False, subset=None, idx=0, prominence=0.1) 
ds12 = parse_auto('./data/paleomon/fd_auto_15_15ppt_medium_F7_68.csv', inv=False, subset=None, idx=1, prominence=0.1) 
ds13 = parse_auto('./data/paleomon/fd_auto_15_15ppt_medium_G2_113.csv', inv=False, subset=None, idx=1, prominence=0.1) 
ds14 = parse_auto('./data/paleomon/fd_auto_15_15ppt_medium_G5_1.csv', inv=False, subset=None, idx=1, prominence=0.1) 
ds15 = parse_auto('./data/paleomon/fd_auto_15_15ppt_medium_G1_82.csv', inv=False, subset=None, idx=1, prominence=0.07) 
# ds16 = parse_auto('./data/paleomon/fd_auto_15_15ppt_young_H12_119.csv', inv=False, subset=None, idx=0, prominence=0.1) 
# ds17 = parse_auto('./data/paleomon/fd_auto_15_15ppt_old_B5_26.csv', inv=False, subset=None, idx=0, prominence=0.1) 
# ds18 = parse_auto('./data/paleomon/fd_auto_15_15ppt_old_B8_1.csv', inv=False, subset=None, idx=1, prominence=0) 
# ds19 = parse_auto('./data/paleomon/fd_auto_15_15ppt_old_D10_42.csv', inv=False, subset=None, idx=1, prominence=0.1) 
# ds20 = parse_auto('./data/paleomon/fd_auto_15_15ppt_old_E5_15.csv', inv=False, subset=None, idx=0, prominence=0.1) 
# ds21 = parse_auto('./data/paleomon/fd_auto_15_15ppt_old_F4_1.csv', inv=False, subset=None, idx=0, prominence=0.1) 
# ds22 = parse_auto('./data/paleomon/fd_auto_15_15ppt_old_F10_21.csv', inv=False, subset=None, idx=0, prominence=0.1) 
# ds23 = parse_auto('./data/paleomon/fd_auto_15_15ppt_old_H6.csv', inv=False, subset=None, idx=0, prominence=0.1) 

# Man
# mds = parse_man('./data/paleomon/hr_man_15_15ppt_medium_1.csv', subset=dict(d=slice(1, None, None), s=slice(1, None, None))) 
# mds1 = parse_man('./data/paleomon/hr_man_15_15ppt_old_1.csv') # s[1:]
mds2 = parse_man('./data/paleomon/hr_man_15_15ppt_young_1.csv') # d[:-1], s[1:]
# mds3 = parse_man('./data/paleomon/hr_man_15_15ppt_young_A1_37.csv')
mds4 = parse_man('./data/paleomon/hr_man_15_15ppt_young_A1_60.csv') 
mds5 = parse_man('./data/paleomon/hr_man_15_15ppt_young_A4_60.csv') # s[1:]
# mds6 = parse_man('./data/paleomon/hr_man_15_15ppt_young_C12_1.csv') # s[1:]
# mds7 = parse_man('./data/paleomon/hr_man_15_15ppt_young_D3_50.csv')  # s[1:]
# mds8 = parse_man('./data/paleomon/hr_man_15_15ppt_young_H3_90.csv') # s[1:]
mds9 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_B2_60.csv')
mds10 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_C2_149.csv')
# mds11 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_C5_1.csv')
mds12 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_F7_68.csv')
mds13 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_G2_113.csv')
mds14 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_G5_1.csv')
mds15 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_G1_82.csv')
# mds16 = parse_man('./data/paleomon/hr_man_15_15ppt_young_H12_119.csv')
# mds17 = parse_man('./data/paleomon/hr_man_15_15ppt_old_B5_26.csv') 
# mds18 = parse_man('./data/paleomon/hr_man_15_15ppt_old_B8_1.csv') 
# mds19 = parse_man('./data/paleomon/hr_man_15_15ppt_old_D10_42.csv') 
# mds20 = parse_man('./data/paleomon/hr_man_15_15ppt_old_E5_15.csv') 
# mds21 = parse_man('./data/paleomon/hr_man_15_15ppt_old_F4_1.csv') 
# mds22 = parse_man('./data/paleomon/hr_man_15_15ppt_old_F10_21.csv') 
# mds23 = parse_man('./data/paleomon/hr_man_15_15ppt_old_H6.csv') 

msv = []
(md,ms),(mad,mas) = mds2[0],mds2[1]
# msv.append(np.mean(scale(mad[:6] - mas[:6])))
(md,ms),(mad,mas) = mds4[0],mds4[1]
msv.append(np.mean(scale(mad[:6] - mas[:6])))
# (md,ms),(mad,mas) = mds5[0],mds5[1]
# msv.append(np.mean(scale(mad[:6] - mas[:6])))
(md,ms),(mad,mas) = mds2[0],mds9[1]
msv.append(np.mean(scale(mad[:6] - mas[:6])))
(md,ms),(mad,mas) = mds4[0],mds10[1]
# msv.append(np.mean(scale(mad[:6] - mas[:6])))
(md,ms),(mad,mas) = mds5[0],mds12[1]
msv.append(np.mean(scale(mad[:6] - mas[:6])))
(md,ms),(mad,mas) = mds5[0],mds13[1]
msv.append(np.mean(scale(mad[:6] - mas[:6])))
(md,ms),(mad,mas) = mds5[0],mds14[1]
msv.append(np.mean(scale(mad[:6] - mas[:6])))
(md,ms),(mad,mas) = mds5[0],mds15[1]
# msv.append(np.mean(scale(mad[:6] - mas[:6])))

asv = []
d,aad,s,aas = ds2
# asv.append(np.mean(scale(aad[:6])))
d,aad,s,aas = ds4
asv.append(np.mean(scale(aad[:6])))
# d,aad,s,aas = ds5
# asv.append(np.mean(scale(aad[:6])))
d,aad,s,aas = ds9
asv.append(np.mean(scale(aad[:6])))
d,aad,s,aas = ds10
# asv.append(np.mean(scale(aad[:6])))
d,aad,s,aas = ds12
asv.append(np.mean(scale(aad[:6])))
d,aad,s,aas = ds13
asv.append(np.mean(scale(aad[:6])))
d,aad,s,aas = ds14
asv.append(np.mean(scale(aad[:6])))
d,aad,s,aas = ds15
# asv.append(np.mean(scale(aad[:6])))


plt.scatter(msv, asv)
plt.show()