import matplotlib.pyplot as plt
import numpy as np
import heartcv as hcv
import pandas as pd
import glob
from statsmodels.nonparametric.smoothers_lowess import lowess

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

def parse_auto(file, inv, subset, frac, plot=False, *args, **kwargs):
    dat = pd.read_csv(file)
    area = dat['area']
    if inv:
        area = np.max(area) - area
    area = scale(area)[:300] # to 300 as manual is only up to 300
    smoothed = lowess(area, dat['frame'][:300], it=0, frac=frac)[:,1]
    if frac == 0:
        smoothed = area
    t,d,s = hcv.find_events(smoothed, *args, **kwargs)

    d = d[0]
    s = s[0]

    if subset:
        d_to = subset.pop('d', None)
        s_to = subset.pop('s', None)

        if d_to:
            if d_to < 0:
                d = d[:d_to]
            else:
                d = d[d_to:]
        if s_to:
            if s_to < 0:
                s = s[:s_to]
            else:
                s = s[s_to:]

    if plot:
        plt.plot(area)
        plt.plot(d, area[d], 'x')
        plt.plot(s, area[s], 'x')
        plt.show()

    print(tuple(map(len, (d,s))))

    return (d,s)

def parse_man(file, subset=None):
    dat = pd.read_csv(file)
    d_peaks = dat['EndDiastoleFrame']
    s_peaks = dat['EndSystoleFrame']

    if subset:
        d_to = subset.pop('d', None)
        s_to = subset.pop('s', None)

        if d_to:
            d_peaks = d_peaks[d_to]
        if s_to:
            s_peaks = s_peaks[s_to]

    return tuple(map(np.asarray, (d_peaks, s_peaks)))

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
    s_time = d_peaks - s_peaks

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
    s_diffs = s_peaks[1:] - s_peaks[:-1] # only use systole as more reliablly accurate
    min_t = s_diffs.min()
    max_t = s_diffs.max()
    mean_t = s_diffs.mean()
    med_t = np.median(s_diffs)
    std_t = s_diffs.std()
    
    # RMSS
    rmss = rmse(s_diffs[1:], s_diffs[:-1])    

    return [[hr], 
            [min_t], [max_t], [mean_t], [med_t], [std_t], [rmss], 
            [min_dt], [max_dt], [mean_dt], [med_dt], [std_dt], 
            [min_st], [max_st], [mean_st], [med_st], [std_st]]

def append_with(to, with_):
    new = []
    for lt,lw in zip(to, with_):
        new.append(lt + lw)
    return new

# # Auto
# ds = parse_auto('./data/paleomon/sv_auto_15_15ppt_medium_1.csv', inv=False, subset=dict(s=1), frac=0, prominence=0.1) 
# ds1 = parse_auto('./data/paleomon/sv_auto_15_15ppt_old_1.csv', inv=False, subset=dict(s=1), frac=0, prominence=0.05, distance=3) # s[1:]
# ds2 = parse_auto('./data/paleomon/sv_auto_15_15ppt_young_1.csv', inv=False, subset=dict(d=-1, s=1), frac=0, prominence=0.2, distance=3) # d[:-1], s[1:]
# ds3 = parse_auto('./data/paleomon/sv_auto_15_15ppt_young_A1_37.csv', inv=True, subset=None, frac=0.015, prominence=0.2)
# ds4 = parse_auto('./data/paleomon/sv_auto_15_15ppt_young_A1_60.csv', inv=True, subset=None, frac=0, prominence=0.2)
# ds5 = parse_auto('./data/paleomon/sv_auto_15_15ppt_young_A4_60.csv', inv=False, subset=dict(s=1), frac=0, prominence=0.4) # s[1:]
# ds6 = parse_auto('./data/paleomon/sv_auto_15_15ppt_young_C12_1.csv', inv=True, subset=dict(s=1), frac=0, prominence=0.4) # s[1:]
# ds7 = parse_auto('./data/paleomon/sv_auto_15_15ppt_young_D3_50.csv', inv=True, subset=None, frac=0, prominence=0.3) # s[1:]
# ds8 = parse_auto('./data/paleomon/sv_auto_15_15ppt_young_H3_90.csv', inv=False, subset=dict(s=1), frac=0, prominence=0.3) # s[1:]
# ds9 = parse_auto('./data/paleomon/sv_auto_15_15ppt_medium_B2_60.csv', inv=False, subset=dict(d=-1), frac=0, prominence=0.3) 
# ds10 = parse_auto('./data/paleomon/sv_auto_15_15ppt_medium_C2_149.csv', inv=False, subset=None, frac=0, prominence=0.1) 
# ds11 = parse_auto('./data/paleomon/sv_auto_15_15ppt_medium_C5_1.csv', inv=True, subset=dict(d=-1), frac=0, prominence=0.09) 
# ds12 = parse_auto('./data/paleomon/sv_auto_15_15ppt_medium_F7_68.csv', inv=True, subset=dict(s=1), frac=0, prominence=0.1) 
# ds13 = parse_auto('./data/paleomon/sv_auto_15_15ppt_medium_G2_113.csv', inv=False, subset=dict(s=1), frac=0, prominence=0.1) 
# ds14 = parse_auto('./data/paleomon/sv_auto_15_15ppt_medium_G5_1.csv', inv=False, subset=dict(d=-1, s=1), frac=0, prominence=0.1) 
# ds15 = parse_auto('./data/paleomon/sv_auto_15_15ppt_medium_G1_82.csv', inv=True, subset=None, frac=0, prominence=0.2, distance=5) 
# ds16 = parse_auto('./data/paleomon/sv_auto_15_15ppt_young_H12_119.csv', inv=False, subset=None, frac=0, plot=True, prominence=0.2) 

# # Man
# mds = parse_man('./data/paleomon/hr_man_15_15ppt_medium_1.csv', subset=dict(d=slice(1, None, None), s=slice(1, None, None))) 
# mds1 = parse_man('./data/paleomon/hr_man_15_15ppt_old_1.csv') # s[1:]
# mds2 = parse_man('./data/paleomon/hr_man_15_15ppt_young_1.csv') # d[:-1], s[1:]
# mds3 = parse_man('./data/paleomon/hr_man_15_15ppt_young_A1_37.csv')
# mds4 = parse_man('./data/paleomon/hr_man_15_15ppt_young_A1_60.csv') 
# mds5 = parse_man('./data/paleomon/hr_man_15_15ppt_young_A4_60.csv') # s[1:]
# mds6 = parse_man('./data/paleomon/hr_man_15_15ppt_young_C12_1.csv') # s[1:]
# mds7 = parse_man('./data/paleomon/hr_man_15_15ppt_young_D3_50.csv')  # s[1:]
# mds8 = parse_man('./data/paleomon/hr_man_15_15ppt_young_H3_90.csv') # s[1:]
# mds9 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_B2_60.csv')
# mds10 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_C2_149.csv')
# mds11 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_C5_1.csv')
# mds12 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_F7_68.csv')
# mds13 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_G2_113.csv')
# mds14 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_G5_1.csv')
# mds15 = parse_man('./data/paleomon/hr_man_15_15ppt_medium_G1_82.csv')
# mds16 = parse_man('./data/paleomon/hr_man_15_15ppt_young_H12_119.csv')

# # Create containers
# a_hr = []
# m_hr = []
# a_min_t = []
# m_min_t = []
# a_max_t = []
# m_max_t = []
# a_mean_t = []
# m_mean_t = []
# a_med_t = []
# m_med_t = []
# a_std_t = []
# m_std_t = []
# a_rmss = []
# m_rmss = []

# a_min_dt = []
# m_min_dt = []
# a_max_dt = []
# m_max_dt = []
# a_mean_dt = []
# m_mean_dt = []
# a_med_dt = []
# m_med_dt = []
# a_std_dt = []
# m_std_dt = []

# a_min_st = []
# m_min_st = []
# a_max_st = []
# m_max_st = []
# a_mean_st = []
# m_mean_st = []
# a_med_st = []
# m_med_st = []
# a_std_st = []
# m_std_st = []

# all_a = [a_hr, a_min_t, a_max_t, a_mean_t, a_med_t, a_std_t, a_rmss, a_min_dt, a_max_dt, a_mean_dt, a_med_dt, a_std_dt, a_min_st, a_max_st, a_mean_st, a_med_st, a_std_st]
# all_m = [m_hr, m_min_t, m_max_t, m_mean_t, m_med_t, m_std_t, m_rmss, m_min_dt, m_max_dt, m_mean_dt, m_med_dt, m_std_dt, m_min_st, m_max_st, m_mean_st, m_med_st, m_std_st]

# # Compute stats (auto)
# all_a = append_with(all_a, stats(*ds))
# all_a = append_with(all_a, stats(*ds1))
# all_a = append_with(all_a, stats(*ds2))
# all_a = append_with(all_a, stats(*ds3))
# all_a = append_with(all_a, stats(*ds4))
# all_a = append_with(all_a, stats(*ds5))
# all_a = append_with(all_a, stats(*ds6))
# all_a = append_with(all_a, stats(*ds7))
# all_a = append_with(all_a, stats(*ds8))
# all_a = append_with(all_a, stats(*ds9))
# all_a = append_with(all_a, stats(*ds10))
# all_a = append_with(all_a, stats(*ds11))
# all_a = append_with(all_a, stats(*ds12))
# all_a = append_with(all_a, stats(*ds13))
# all_a = append_with(all_a, stats(*ds14))
# all_a = append_with(all_a, stats(*ds15))
# all_a = append_with(all_a, stats(*ds16))

# # Compute stats (man)
# all_m = append_with(all_m, stats(*mds))
# all_m = append_with(all_m, stats(*mds1))
# all_m = append_with(all_m, stats(*mds2))
# all_m = append_with(all_m, stats(*mds3))
# all_m = append_with(all_m, stats(*mds4))
# all_m = append_with(all_m, stats(*mds5))
# all_m = append_with(all_m, stats(*mds6))
# all_m = append_with(all_m, stats(*mds7))
# all_m = append_with(all_m, stats(*mds8))
# all_m = append_with(all_m, stats(*mds9))
# all_m = append_with(all_m, stats(*mds10))
# all_m = append_with(all_m, stats(*mds11))
# all_m = append_with(all_m, stats(*mds12))
# all_m = append_with(all_m, stats(*mds13))
# all_m = append_with(all_m, stats(*mds14))
# all_m = append_with(all_m, stats(*mds15))
# all_m = append_with(all_m, stats(*mds16))

# # Deconstruct
# [a_hr, a_min_t, a_max_t, a_mean_t, a_med_t, a_std_t, a_rmss, a_min_dt, a_max_dt, a_mean_dt, a_med_dt, a_std_dt, a_min_st, a_max_st, a_mean_st, a_med_st, a_std_st] = all_a
# [m_hr, m_min_t, m_max_t, m_mean_t, m_med_t, m_std_t, m_rmss, m_min_dt, m_max_dt, m_mean_dt, m_med_dt, m_std_dt, m_min_st, m_max_st, m_mean_st, m_med_st, m_std_st] = all_m

# # And plot
# plt.scatter(m_hr, a_hr)
# plt.xlabel('Manual HR (for 10s)')
# plt.ylabel('HeartCV HR (for 10s)')
# plt.show()

# # Beat to beat stats
# fig, ((ax1,ax2,ax3,),(ax4,ax5,ax6)) = plt.subplots(2, 3)
# ax1.scatter(m_min_t, a_min_t)
# ax1.set_xlabel('Manual min beat to beat timing')
# ax1.set_ylabel('HeartCV min beat to beat timing')

# ax2.scatter(m_max_t, a_max_t)
# ax2.set_xlabel('Manual max beat to beat timing')
# ax2.set_ylabel('HeartCV max beat to beat timing')

# ax3.scatter(m_mean_t, a_mean_t)
# ax3.set_xlabel('Manual mean beat to beat timing')
# ax3.set_ylabel('HeartCV mean beat to beat timing')

# ax4.scatter(m_med_t, a_med_t)
# ax4.set_xlabel('Manual med beat to beat timing')
# ax4.set_ylabel('HeartCV med beat to beat timing')

# ax5.scatter(m_std_t, a_std_t)
# ax5.set_xlabel('Manual std beat to beat timing')
# ax5.set_ylabel('HeartCV std beat to beat timing')

# ax6.scatter(m_rmss, a_rmss)
# ax6.set_xlabel('Manual rmss beat to beat timing')
# ax6.set_ylabel('HeartCV rmss beat to beat timing')
# plt.show()

# # Diastole timing stats
# fig, ((ax1,ax2,ax3,),(ax4,ax5,ax6)) = plt.subplots(2, 3)
# ax1.scatter(m_min_dt, a_min_dt)
# ax1.set_xlabel('Manual min diastole timing')
# ax1.set_ylabel('HeartCV min diastole timing')

# ax2.scatter(m_max_dt, a_max_dt)
# ax2.set_xlabel('Manual max diastole timing')
# ax2.set_ylabel('HeartCV max diastole timing')

# ax3.scatter(m_mean_dt, a_mean_dt)
# ax3.set_xlabel('Manual mean diastole timing')
# ax3.set_ylabel('HeartCV mean diastole timing')

# ax4.scatter(m_med_dt, a_med_dt)
# ax4.set_xlabel('Manual med diastole timing')
# ax4.set_ylabel('HeartCV med diastole timing')

# ax5.scatter(m_std_dt, a_std_dt)
# ax5.set_xlabel('Manual std diastole timing')
# ax5.set_ylabel('HeartCV std diastole timing')

# plt.show()

# # Systole timing stats
# fig, ((ax1,ax2,ax3,),(ax4,ax5,ax6)) = plt.subplots(2, 3)
# ax1.scatter(m_min_st, a_min_st)
# ax1.set_xlabel('Manual min systole timing')
# ax1.set_ylabel('HeartCV min systole timing')

# ax2.scatter(m_max_st, a_max_st)
# ax2.set_xlabel('Manual max systole timing')
# ax2.set_ylabel('HeartCV max systole timing')

# ax3.scatter(m_mean_st, a_mean_st)
# ax3.set_xlabel('Manual mean systole timing')
# ax3.set_ylabel('HeartCV mean systole timing')

# ax4.scatter(m_med_st, a_med_st)
# ax4.set_xlabel('Manual med systole timing')
# ax4.set_ylabel('HeartCV med systole timing')

# ax5.scatter(m_std_st, a_std_st)
# ax5.set_xlabel('Manual std systole timing')
# ax5.set_ylabel('HeartCV std systole timing')

# plt.show()


# Auto
ds = parse_auto('./data/radix/mpx_auto_20deg_A1.csv', inv=False, subset=None, frac=0, prominence=0.1) 
ds1 = parse_auto('./data/radix/mpx_auto_20deg_A3.csv', inv=False, subset=dict(s=1), frac=0, prominence=0.1) # s[1:]
ds2 = parse_auto('./data/radix/mpx_auto_20deg_A4.csv', inv=False, subset=dict(s=1), frac=0, prominence=0.2) # d[:-1], s[1:]
ds3 = parse_auto('./data/radix/mpx_auto_20deg_A6.csv', inv=False, subset=dict(d=-1), frac=0, prominence=0.2)
ds4 = parse_auto('./data/radix/mpx_auto_20deg_B1.csv', inv=False, subset=None, frac=0, prominence=0.1)
ds5 = parse_auto('./data/radix/mpx_auto_20deg_B2.csv', inv=True, subset=None, frac=0, prominence=0.05, distance=8) # s[1:]
ds6 = parse_auto('./data/radix/mpx_auto_20deg_B3.csv', inv=True, subset=dict(d=-1, s=1), frac=0, prominence=0.14) # s[1:]b
ds7 = parse_auto('./data/radix/mpx_auto_25deg_A1.csv', inv=True, subset=dict(d=-1), frac=0, prominence=0.05) # s[1:]
ds8 = parse_auto('./data/radix/mpx_auto_25deg_A4.csv', inv=True, subset=dict(d=-1, s=1), frac=0, prominence=0.1) # s[1:]
ds9 = parse_auto('./data/radix/mpx_auto_25deg_A5.csv', inv=False, subset=dict(d=-1, s=1), frac=0, prominence=0.1)
ds10 = parse_auto('./data/radix/mpx_auto_25deg_A6.csv', inv=False, subset=dict(d=-1), frac=0, prominence=0.04, distance=5)
ds11 = parse_auto('./data/radix/mpx_auto_25deg_A7.csv', inv=True, subset=None, frac=0, prominence=0.1)
ds12 = parse_auto('./data/radix/mpx_auto_25deg_B1.csv', inv=True, subset=dict(d=-1), frac=0, prominence=0.1)
# ds13 = parse_auto('./data/radix/mpx_auto_25deg_B2.csv', inv=True, subset=dict(d=-1), frac=0, prominence=0.1)
# ds14 = parse_auto('./data/radix/mpx_auto_25deg_B3.csv', inv=False, subset=dict(d=-1), frac=0, prominence=0.08, distance=5)
ds13 = parse_auto('./data/radix/mpx_auto_30deg_A1.csv', inv=False, subset=dict(d=-2, s=-1), frac=0, prominence=0.08)
ds14 = parse_auto('./data/radix/mpx_auto_30deg_A4.csv', inv=False, subset=dict(d=-1), frac=0, prominence=0.001)
ds15 = parse_auto('./data/radix/mpx_auto_30deg_A5.csv', inv=False, subset=dict(s=1), frac=0, prominence=0.01)
# ds16 = parse_auto('./data/radix/mpx_auto_30deg_A6.csv', inv=False, subset=None, frac=0, prominence=0.05)
# ds16 = parse_auto('./data/radix/mpx_auto_30deg_A7.csv', inv=False, subset=None, frac=0, prominence=0.1)
ds16 = parse_auto('./data/radix/mpx_auto_30deg_B1.csv', inv=False, subset=dict(d=-1), frac=0, prominence=0.01) 
ds17 = parse_auto('./data/radix/mpx_auto_30deg_B3.csv', inv=True, subset=dict(s=1), frac=0, prominence=0.04) 
ds18 = parse_auto('./data/radix/mpx_auto_30deg_B5.csv', inv=False, subset=None, frac=0, prominence=0.15, distance=5) 

# Man
mds = parse_man('./data/radix/hr_man_20deg_A1.csv') 
mds1 = parse_man('./data/radix/hr_man_20deg_A3.csv') # s[1:]
mds2 = parse_man('./data/radix/hr_man_20deg_A4.csv') # d[:-1]s[1:]
mds3 = parse_man('./data/radix/hr_man_20deg_A6.csv')
mds4 = parse_man('./data/radix/hr_man_20deg_B1.csv')
mds5 = parse_man('./data/radix/hr_man_20deg_B2.csv') # s[1:]
mds6 = parse_man('./data/radix/hr_man_20deg_B3.csv') # s[1:]b
mds7 = parse_man('./data/radix/hr_man_25deg_A1.csv') # s[1:]
mds8 = parse_man('./data/radix/hr_man_25deg_A4.csv') # s[1:]
mds9 = parse_man('./data/radix/hr_man_25deg_A5.csv')
mds10 = parse_man('./data/radix/hr_man_25deg_A6.csv')
mds11 = parse_man('./data/radix/hr_man_25deg_A7.csv')
mds12 = parse_man('./data/radix/hr_man_25deg_B1.csv')
# mds13 = parse_man('./data/radix/hr_man_25deg_B2.csv')
# mds14 = parse_man('./data/radix/hr_man_25deg_B3.csv')
mds13 = parse_man('./data/radix/hr_man_30deg_A1.csv')
mds14 = parse_man('./data/radix/hr_man_30deg_A4.csv')
mds15 = parse_man('./data/radix/hr_man_30deg_A5.csv')
# mds16 = parse_man('./data/radix/hr_man_30deg_A6.csv')
# mds16 = parse_man('./data/radix/hr_man_30deg_A7.csv')
mds16 = parse_man('./data/radix/hr_man_30deg_B1.csv') 
mds17 = parse_man('./data/radix/hr_man_30deg_B3.csv') 
mds18 = parse_man('./data/radix/hr_man_30deg_B5.csv') 

# Create containers
a_hr = []
m_hr = []
a_min_t = []
m_min_t = []
a_max_t = []
m_max_t = []
a_mean_t = []
m_mean_t = []
a_med_t = []
m_med_t = []
a_std_t = []
m_std_t = []
a_rmss = []
m_rmss = []

a_min_dt = []
m_min_dt = []
a_max_dt = []
m_max_dt = []
a_mean_dt = []
m_mean_dt = []
a_med_dt = []
m_med_dt = []
a_std_dt = []
m_std_dt = []

a_min_st = []
m_min_st = []
a_max_st = []
m_max_st = []
a_mean_st = []
m_mean_st = []
a_med_st = []
m_med_st = []
a_std_st = []
m_std_st = []

all_a = [a_hr, a_min_t, a_max_t, a_mean_t, a_med_t, a_std_t, a_rmss, a_min_dt, a_max_dt, a_mean_dt, a_med_dt, a_std_dt, a_min_st, a_max_st, a_mean_st, a_med_st, a_std_st]
all_m = [m_hr, m_min_t, m_max_t, m_mean_t, m_med_t, m_std_t, m_rmss, m_min_dt, m_max_dt, m_mean_dt, m_med_dt, m_std_dt, m_min_st, m_max_st, m_mean_st, m_med_st, m_std_st]


# Compute stats (auto)
all_a = append_with(all_a, stats(*ds))
all_a = append_with(all_a, stats(*ds1))
all_a = append_with(all_a, stats(*ds2))
all_a = append_with(all_a, stats(*ds3))
all_a = append_with(all_a, stats(*ds4))
all_a = append_with(all_a, stats(*ds5))
all_a = append_with(all_a, stats(*ds6))
all_a = append_with(all_a, stats(*ds7))
all_a = append_with(all_a, stats(*ds8))
all_a = append_with(all_a, stats(*ds9))
# all_a = append_with(all_a, stats(*ds10))
all_a = append_with(all_a, stats(*ds11))
all_a = append_with(all_a, stats(*ds12))
# all_a = append_with(all_a, stats(*ds13))
all_a = append_with(all_a, stats(*ds14))
all_a = append_with(all_a, stats(*ds15))
all_a = append_with(all_a, stats(*ds16))
all_a = append_with(all_a, stats(*ds17))
all_a = append_with(all_a, stats(*ds18))

# Compute stats (man)
all_m = append_with(all_m, stats(*mds))
all_m = append_with(all_m, stats(*mds1))
all_m = append_with(all_m, stats(*mds2))
all_m = append_with(all_m, stats(*mds3))
all_m = append_with(all_m, stats(*mds4))
all_m = append_with(all_m, stats(*mds5))
all_m = append_with(all_m, stats(*mds6))
all_m = append_with(all_m, stats(*mds7))
all_m = append_with(all_m, stats(*mds8))
all_m = append_with(all_m, stats(*mds9))
# all_m = append_with(all_m, stats(*mds10))
all_m = append_with(all_m, stats(*mds11))
all_m = append_with(all_m, stats(*mds12))
# all_m = append_with(all_m, stats(*mds13))
all_m = append_with(all_m, stats(*mds14))
all_m = append_with(all_m, stats(*mds15))
all_m = append_with(all_m, stats(*mds16))
all_m = append_with(all_m, stats(*mds17))
all_m = append_with(all_m, stats(*mds18))

# Deconstruct
[a_hr, a_min_t, a_max_t, a_mean_t, a_med_t, a_std_t, a_rmss, a_min_dt, a_max_dt, a_mean_dt, a_med_dt, a_std_dt, a_min_st, a_max_st, a_mean_st, a_med_st, a_std_st] = all_a
[m_hr, m_min_t, m_max_t, m_mean_t, m_med_t, m_std_t, m_rmss, m_min_dt, m_max_dt, m_mean_dt, m_med_dt, m_std_dt, m_min_st, m_max_st, m_mean_st, m_med_st, m_std_st] = all_m

# # And plot
# plt.scatter(m_hr, a_hr)
# plt.xlabel('Manual HR (for 10s)')
# plt.ylabel('HeartCV HR (for 10s)')
# plt.show()

# Beat to beat stats
fig, ((ax1,ax2,ax3,),(ax4,ax5,ax6)) = plt.subplots(2, 3)
ax1.scatter(m_min_t, a_min_t)
ax1.set_xlabel('Manual min beat to beat timing')
ax1.set_ylabel('HeartCV min beat to beat timing')

ax2.scatter(m_max_t, a_max_t)
ax2.set_xlabel('Manual max beat to beat timing')
ax2.set_ylabel('HeartCV max beat to beat timing')

ax3.scatter(m_mean_t, a_mean_t)
ax3.set_xlabel('Manual mean beat to beat timing')
ax3.set_ylabel('HeartCV mean beat to beat timing')

ax4.scatter(m_med_t, a_med_t)
ax4.set_xlabel('Manual med beat to beat timing')
ax4.set_ylabel('HeartCV med beat to beat timing')

ax5.scatter(m_std_t, a_std_t)
ax5.set_xlabel('Manual std beat to beat timing')
ax5.set_ylabel('HeartCV std beat to beat timing')

ax6.scatter(m_rmss, a_rmss)
ax6.set_xlabel('Manual rmss beat to beat timing')
ax6.set_ylabel('HeartCV rmss beat to beat timing')
plt.show()

# # Diastole timing stats
# fig, ((ax1,ax2,ax3,),(ax4,ax5,ax6)) = plt.subplots(2, 3)
# ax1.scatter(m_min_dt, a_min_dt)
# ax1.set_xlabel('Manual min diastole timing')
# ax1.set_ylabel('HeartCV min diastole timing')

# ax2.scatter(m_max_dt, a_max_dt)
# ax2.set_xlabel('Manual max diastole timing')
# ax2.set_ylabel('HeartCV max diastole timing')

# ax3.scatter(m_mean_dt, a_mean_dt)
# ax3.set_xlabel('Manual mean diastole timing')
# ax3.set_ylabel('HeartCV mean diastole timing')

# ax4.scatter(m_med_dt, a_med_dt)
# ax4.set_xlabel('Manual med diastole timing')
# ax4.set_ylabel('HeartCV med diastole timing')

# ax5.scatter(m_std_dt, a_std_dt)
# ax5.set_xlabel('Manual std diastole timing')
# ax5.set_ylabel('HeartCV std diastole timing')

# plt.show()

# # Systole timing stats
# fig, ((ax1,ax2,ax3,),(ax4,ax5,ax6)) = plt.subplots(2, 3)
# ax1.scatter(m_min_st, a_min_st)
# ax1.set_xlabel('Manual min systole timing')
# ax1.set_ylabel('HeartCV min systole timing')

# ax2.scatter(m_max_st, a_max_st)
# ax2.set_xlabel('Manual max systole timing')
# ax2.set_ylabel('HeartCV max systole timing')

# ax3.scatter(m_mean_st, a_mean_st)
# ax3.set_xlabel('Manual mean systole timing')
# ax3.set_ylabel('HeartCV mean systole timing')

# ax4.scatter(m_med_st, a_med_st)
# ax4.set_xlabel('Manual med systole timing')
# ax4.set_ylabel('HeartCV med systole timing')

# ax5.scatter(m_std_st, a_std_st)
# ax5.set_xlabel('Manual std systole timing')
# ax5.set_ylabel('HeartCV std systole timing')

# plt.show()