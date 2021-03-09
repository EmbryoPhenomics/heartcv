import matplotlib.pyplot as plt
import numpy as np
import heartcv as hcv
import pandas as pd
import glob
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr

from heartcv import minmax_scale as scale

def HR(hr_f, l, fs):
    return (hr_f/(l/fs))*60

def HRV(in_stat, fs):
    return in_stat/fs

def corr_stats(r, p):
    if p <= 0.05:
        if p <= 0.01:
            if p <= 0.001:
                if p <= 0.0001:
                    p_str = 'p < 0.0001'
                else:
                    p_str = 'p < 0.001'
            else:
                p_str = 'p < 0.01'
        else:
            p_str = 'p < 0.05'
    else:
        p_str = 'p > 0.05'

    r_str = f'R = {round(r, 3)}'

    return r_str, p_str

def mse(truth, act):
    truth, act = map(np.asarray, (truth, act))
    truth, act = map(scale, (truth, act))
    diff = truth - act
    return np.nanmean(diff ** 2)


def rmse(truth, act):
    return np.sqrt(mse(truth, act))


def between(arr, from_, to_):
    at = []
    for f, t in zip(from_, to_):
        inds = list(range(f, t + 1))
        ret = arr[slice(f, t + 1, None)]
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
    area = dat["area"]
    if inv:
        area = np.max(area) - area
    area = scale(area)[:300]  # to 300 as manual is only up to 300
    smoothed = lowess(area, dat["frame"][:300], it=0, frac=frac)[:, 1]
    if frac == 0:
        smoothed = area
    t, d, s = hcv.find_events(smoothed, *args, **kwargs)

    d = d[0]
    s = s[0]

    if subset:
        d_to = subset.pop("d", None)
        s_to = subset.pop("s", None)

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
        plt.plot(d, area[d], "x")
        plt.plot(s, area[s], "x")
        # plt.title(file)
        plt.show()

    print(tuple(map(len, (d, s))))

    return (d, s)


def parse_man(file, subset=None):
    dat = pd.read_csv(file)
    d_peaks = dat["EndDiastoleFrame"]
    d_peaks = d_peaks[d_peaks <= 300]
    s_peaks = dat["EndSystoleFrame"]
    s_peaks = s_peaks[s_peaks <= 300]

    if subset:
        d_to = subset.pop("d", None)
        s_to = subset.pop("s", None)

        if d_to:
            d_peaks = d_peaks[d_to]
        if s_to:
            s_peaks = s_peaks[s_to]

    return tuple(map(np.asarray, (s_peaks, d_peaks)))


def stats(d_peaks, s_peaks, l, fs):
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

    hr = np.mean((len(s_peaks), len(d_peaks)))
    hr = HR(hr, l, fs)

    # Diffs
    d_diffs = d_peaks[1:] - d_peaks[:-1]
    d_diffs = HRV(d_diffs, fs)
    min_t = d_diffs.min()
    max_t = d_diffs.max()
    mean_t = d_diffs.mean()
    med_t = np.median(d_diffs)
    std_t = d_diffs.std()

    # RMSS
    rmssd = rmse(d_diffs[1:], d_diffs[:-1])

    # Diffs
    s_diffs = s_peaks[1:] - s_peaks[:-1]
    s_diffs = HRV(s_diffs, fs)
    min_t = np.mean((min_t, s_diffs.min()))
    max_t = np.mean((max_t, s_diffs.min()))
    mean_t = np.mean((mean_t, s_diffs.min()))
    med_t = np.mean((med_t, s_diffs.min()))
    std_t = np.mean((std_t, s_diffs.min()))

    # RMSS
    rmssd = np.mean((rmssd, rmse(s_diffs[1:], s_diffs[:-1])))

    return [[hr], [min_t], [max_t], [mean_t], [med_t], [std_t], [rmssd]]


def append_with(to, with_):
    new = []
    for lt, lw in zip(to, with_):
        new.append(lt + lw)
    return new


# Auto
ds2 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_young_1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
)  # d[:-1], s[1:]
ds3 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_young_A1_37.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.2,
)
ds4 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_young_A1_60.csv",
    inv=False,
    subset=dict(d=1),
    frac=0,
    prominence=0.3,
)
ds5 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_young_A4_60.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
)  # s[1:]
ds6 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_young_C12_1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
    distance=10,
)  # s[1:]
ds7 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_young_D3_50.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
)  # s[1:]
ds8 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_young_H3_90.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.25,
)  # s[1:]
ds9 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_medium_B2_60.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.2,
    distance=10,
)
ds10 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_medium_C2_149.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
)
ds11 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_medium_C5_1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.2,
)
ds12 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_medium_F7_68.csv",
    inv=False,
    subset=dict(d=-1),
    frac=0,
    prominence=0.3,
    distance=10,
)
ds13 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_medium_G2_113.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.2,
)
ds14 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_medium_G5_1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
)
ds15 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_medium_G1_82.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.3,
    distance=7,
)
ds16 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_young_H12_119.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.4,
)
ds17 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_old_B5_26.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.2,
)
ds18 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_old_B8_1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.3,
    distance=5,
)
ds19 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_old_D10_42.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.3,
)
ds20 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_old_E5_15.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.15,
)
ds21 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_old_F4_1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
)
ds22 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_old_F10_21.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
)
ds23 = parse_auto(
    "./data/paleomon/mpx_auto_15_15ppt_old_H6.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.3,
    distance=7,
)

# Man
mds2 = parse_man("./data/paleomon/hr_man_15_15ppt_young_1.csv")  # d[:-1], s[1:]
mds3 = parse_man("./data/paleomon/hr_man_15_15ppt_young_A1_37.csv")
mds4 = parse_man("./data/paleomon/hr_man_15_15ppt_young_A1_60.csv")
mds5 = parse_man("./data/paleomon/hr_man_15_15ppt_young_A4_60.csv")  # s[1:]
mds6 = parse_man("./data/paleomon/hr_man_15_15ppt_young_C12_1.csv")  # s[1:]
mds7 = parse_man("./data/paleomon/hr_man_15_15ppt_young_D3_50.csv")  # s[1:]
mds8 = parse_man("./data/paleomon/hr_man_15_15ppt_young_H3_90.csv")  # s[1:]
mds9 = parse_man("./data/paleomon/hr_man_15_15ppt_medium_B2_60.csv")
mds10 = parse_man("./data/paleomon/hr_man_15_15ppt_medium_C2_149.csv")
mds11 = parse_man("./data/paleomon/hr_man_15_15ppt_medium_C5_1.csv")
mds12 = parse_man("./data/paleomon/hr_man_15_15ppt_medium_F7_68.csv")
mds13 = parse_man("./data/paleomon/hr_man_15_15ppt_medium_G2_113.csv")
mds14 = parse_man("./data/paleomon/hr_man_15_15ppt_medium_G5_1.csv")
mds15 = parse_man("./data/paleomon/hr_man_15_15ppt_medium_G1_82.csv")
mds16 = parse_man("./data/paleomon/hr_man_15_15ppt_young_H12_119.csv")
mds17 = parse_man("./data/paleomon/hr_man_15_15ppt_old_B5_26.csv")
mds18 = parse_man("./data/paleomon/hr_man_15_15ppt_old_B8_1.csv")
mds19 = parse_man("./data/paleomon/hr_man_15_15ppt_old_D10_42.csv")
mds20 = parse_man("./data/paleomon/hr_man_15_15ppt_old_E5_15.csv")
mds21 = parse_man("./data/paleomon/hr_man_15_15ppt_old_F4_1.csv")
mds22 = parse_man("./data/paleomon/hr_man_15_15ppt_old_F10_21.csv")
mds23 = parse_man("./data/paleomon/hr_man_15_15ppt_old_H6.csv")

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

all_a = [a_hr, a_min_t, a_max_t, a_mean_t, a_med_t, a_std_t, a_rmss]
all_m = [m_hr, m_min_t, m_max_t, m_mean_t, m_med_t, m_std_t, m_rmss]

# Compute stats (auto)
all_a = append_with(all_a, stats(*ds2, 300, 30))
all_a = append_with(all_a, stats(*ds3, 300, 30))
all_a = append_with(all_a, stats(*ds4, 300, 30))
all_a = append_with(all_a, stats(*ds5, 300, 30))
all_a = append_with(all_a, stats(*ds6, 300, 30))
all_a = append_with(all_a, stats(*ds7, 300, 30))
all_a = append_with(all_a, stats(*ds8, 300, 30))
all_a = append_with(all_a, stats(*ds9, 300, 30))
all_a = append_with(all_a, stats(*ds10, 300, 30))
all_a = append_with(all_a, stats(*ds11, 300, 30))
all_a = append_with(all_a, stats(*ds12, 300, 30))
all_a = append_with(all_a, stats(*ds13, 300, 30))
all_a = append_with(all_a, stats(*ds14, 300, 30))
all_a = append_with(all_a, stats(*ds15, 300, 30))
all_a = append_with(all_a, stats(*ds16, 300, 30))
all_a = append_with(all_a, stats(*ds17, 300, 30))
all_a = append_with(all_a, stats(*ds18, 300, 30))
all_a = append_with(all_a, stats(*ds19, 300, 30))
all_a = append_with(all_a, stats(*ds20, 300, 30))
all_a = append_with(all_a, stats(*ds21, 300, 30))
all_a = append_with(all_a, stats(*ds22, 300, 30))
all_a = append_with(all_a, stats(*ds23, 300, 30))

# Compute stats (man)
all_m = append_with(all_m, stats(*mds2, 300, 30))
all_m = append_with(all_m, stats(*mds3, 300, 30))
all_m = append_with(all_m, stats(*mds4, 300, 30))
all_m = append_with(all_m, stats(*mds5, 300, 30))
all_m = append_with(all_m, stats(*mds6, 300, 30))
all_m = append_with(all_m, stats(*mds7, 300, 30))
all_m = append_with(all_m, stats(*mds8, 300, 30))
all_m = append_with(all_m, stats(*mds9, 300, 30))
all_m = append_with(all_m, stats(*mds10, 300, 30))
all_m = append_with(all_m, stats(*mds11, 300, 30))
all_m = append_with(all_m, stats(*mds12, 300, 30))
all_m = append_with(all_m, stats(*mds13, 300, 30))
all_m = append_with(all_m, stats(*mds14, 300, 30))
all_m = append_with(all_m, stats(*mds15, 300, 30))
all_m = append_with(all_m, stats(*mds16, 300, 30))
all_m = append_with(all_m, stats(*mds17, 300, 30))
all_m = append_with(all_m, stats(*mds18, 300, 30))
all_m = append_with(all_m, stats(*mds19, 300, 30))
all_m = append_with(all_m, stats(*mds20, 300, 30))
all_m = append_with(all_m, stats(*mds21, 300, 30))
all_m = append_with(all_m, stats(*mds22, 300, 30))
all_m = append_with(all_m, stats(*mds23, 300, 30))

# Deconstruct
[a_hr, a_min_t, a_max_t, a_mean_t, a_med_t, a_std_t, a_rmss] = all_a
[m_hr, m_min_t, m_max_t, m_mean_t, m_med_t, m_std_t, m_rmss] = all_m

# Auto
ds = parse_auto(
    "./data/radix/mpx_auto_20deg_A1.csv", inv=False, subset=None, frac=0, prominence=0.1
)
ds1 = parse_auto(
    "./data/radix/mpx_auto_20deg_A3.csv", inv=False, subset=None, frac=0, prominence=0.1
)  # s[1:]
ds2 = parse_auto(
    "./data/radix/mpx_auto_20deg_A4.csv", inv=False, subset=None, frac=0, prominence=0.2
)  # d[:-1], s[1:]
ds3 = parse_auto(
    "./data/radix/mpx_auto_20deg_A6.csv", inv=False, subset=None, frac=0, prominence=0.2
)
ds4 = parse_auto(
    "./data/radix/mpx_auto_20deg_B1.csv", inv=False, subset=None, frac=0, prominence=0.1
)
ds5 = parse_auto(
    "./data/radix/mpx_auto_20deg_B2.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.05,
    distance=8,
)  # s[1:]
ds6 = parse_auto(
    "./data/radix/mpx_auto_20deg_B3.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.14,
)  # s[1:]b
ds7 = parse_auto(
    "./data/radix/mpx_auto_25deg_A1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.05,
)  # s[1:]
ds8 = parse_auto(
    "./data/radix/mpx_auto_25deg_A4.csv", inv=False, subset=None, frac=0, prominence=0.1
)  # s[1:]
ds9 = parse_auto(
    "./data/radix/mpx_auto_25deg_A5.csv", inv=False, subset=None, frac=0, prominence=0.1
)
ds11 = parse_auto(
    "./data/radix/mpx_auto_25deg_A7.csv", inv=False, subset=None, frac=0, prominence=0.1
)
ds12 = parse_auto(
    "./data/radix/mpx_auto_25deg_B1.csv", inv=False, subset=None, frac=0, prominence=0.1
)
ds13 = parse_auto(
    "./data/radix/mpx_auto_30deg_A1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.08,
)
ds15 = parse_auto(
    "./data/radix/mpx_auto_30deg_A5.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.01,
)
ds16 = parse_auto(
    "./data/radix/mpx_auto_30deg_B1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.01,
)
ds18 = parse_auto(
    "./data/radix/mpx_auto_30deg_B5.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.15,
    distance=5,
)
ds19 = parse_auto(
    "./data/radix/mpx_auto_unknown_C1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.1,
)
ds21 = parse_auto(
    "./data/radix/mpx_auto_unknown_C3.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.02,
)
ds22 = parse_auto(
    "./data/radix/mpx_auto_unknown_C8.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.02,
    distance=10,
)
ds23 = parse_auto(
    "./data/radix/mpx_auto_unknown_D1.csv",
    inv=False,
    subset=None,
    frac=0,
    prominence=0.15,
)

# Man
mds = parse_man("./data/radix/hr_man_20deg_A1.csv")
mds1 = parse_man("./data/radix/hr_man_20deg_A3.csv")  # s[1:]
mds2 = parse_man("./data/radix/hr_man_20deg_A4.csv")  # d[:-1]s[1:]
mds3 = parse_man("./data/radix/hr_man_20deg_A6.csv")
mds4 = parse_man("./data/radix/hr_man_20deg_B1.csv")
mds5 = parse_man("./data/radix/hr_man_20deg_B2.csv")  # s[1:]
mds6 = parse_man("./data/radix/hr_man_20deg_B3.csv")  # s[1:]b
mds7 = parse_man("./data/radix/hr_man_25deg_A1.csv")  # s[1:]
mds8 = parse_man("./data/radix/hr_man_25deg_A4.csv")  # s[1:]
mds9 = parse_man("./data/radix/hr_man_25deg_A5.csv")
mds11 = parse_man("./data/radix/hr_man_25deg_A7.csv")
mds12 = parse_man("./data/radix/hr_man_25deg_B1.csv")
mds13 = parse_man("./data/radix/hr_man_30deg_A1.csv")
mds15 = parse_man("./data/radix/hr_man_30deg_A5.csv")
mds16 = parse_man("./data/radix/hr_man_30deg_B1.csv")
mds18 = parse_man("./data/radix/hr_man_30deg_B5.csv")
mds19 = parse_man("./data/radix/hr_man_unknown_C1.csv")
mds21 = parse_man("./data/radix/hr_man_unknown_C3.csv")
mds22 = parse_man("./data/radix/hr_man_unknown_C8.csv")
mds23 = parse_man("./data/radix/hr_man_unknown_D1.csv")

# Create containers
a_hr1 = []
m_hr1 = []
a_min_t1 = []
m_min_t1 = []
a_max_t1 = []
m_max_t1 = []
a_mean_t1 = []
m_mean_t1 = []
a_med_t1 = []
m_med_t1 = []
a_std_t1 = []
m_std_t1 = []
a_rmss = []
m_rmss = []

all_a = [a_hr1, a_min_t1, a_max_t1, a_mean_t1, a_med_t1, a_std_t1, a_rmss]
all_m = [m_hr1, m_min_t1, m_max_t1, m_mean_t1, m_med_t1, m_std_t1, m_rmss]

# Compute stats (auto)
all_a = append_with(all_a, stats(*ds, 300, 30))
all_a = append_with(all_a, stats(*ds1, 300, 30))
all_a = append_with(all_a, stats(*ds2, 300, 30))
all_a = append_with(all_a, stats(*ds3, 300, 30))
all_a = append_with(all_a, stats(*ds4, 300, 30))
all_a = append_with(all_a, stats(*ds5, 300, 30))
all_a = append_with(all_a, stats(*ds6, 300, 30))
all_a = append_with(all_a, stats(*ds7, 300, 30))
all_a = append_with(all_a, stats(*ds8, 300, 30))
all_a = append_with(all_a, stats(*ds9, 300, 30))
all_a = append_with(all_a, stats(*ds11, 300, 30))
all_a = append_with(all_a, stats(*ds12, 300, 30))
all_a = append_with(all_a, stats(*ds15, 300, 30))
all_a = append_with(all_a, stats(*ds16, 300, 30))
all_a = append_with(all_a, stats(*ds18, 300, 30))
all_a = append_with(all_a, stats(*ds19, 300, 30))
all_a = append_with(all_a, stats(*ds21, 300, 30))
all_a = append_with(all_a, stats(*ds22, 300, 30))
all_a = append_with(all_a, stats(*ds23, 300, 30))

# Compute stats (man)
all_m = append_with(all_m, stats(*mds, 300, 30))
all_m = append_with(all_m, stats(*mds1, 300, 30))
all_m = append_with(all_m, stats(*mds2, 300, 30))
all_m = append_with(all_m, stats(*mds3, 300, 30))
all_m = append_with(all_m, stats(*mds4, 300, 30))
all_m = append_with(all_m, stats(*mds5, 300, 30))
all_m = append_with(all_m, stats(*mds6, 300, 30))
all_m = append_with(all_m, stats(*mds7, 300, 30))
all_m = append_with(all_m, stats(*mds8, 300, 30))
all_m = append_with(all_m, stats(*mds9, 300, 30))
all_m = append_with(all_m, stats(*mds11, 300, 30))
all_m = append_with(all_m, stats(*mds12, 300, 30))
all_m = append_with(all_m, stats(*mds15, 300, 30))
all_m = append_with(all_m, stats(*mds16, 300, 30))
all_m = append_with(all_m, stats(*mds18, 300, 30))
all_m = append_with(all_m, stats(*mds19, 300, 30))
all_m = append_with(all_m, stats(*mds21, 300, 30))
all_m = append_with(all_m, stats(*mds22, 300, 30))
all_m = append_with(all_m, stats(*mds23, 300, 30))

# Deconstruct
[a_hr1, a_min_t1, a_max_t1, a_mean_t1, a_med_t1, a_std_t1, a_rmss1] = all_a
[m_hr1, m_min_t1, m_max_t1, m_mean_t1, m_med_t1, m_std_t1, m_rmss1] = all_m

# Auto
ds = parse_auto(
    "./data/ciona/mpx_auto_MAH03784.csv", inv=False, subset=None, frac=0, prominence=0.2
)
ds1 = parse_auto(
    "./data/ciona/mpx_auto_MAH03785.csv", inv=False, subset=None, frac=0, prominence=0.2
)
ds2 = parse_auto(
    "./data/ciona/mpx_auto_MAH03786.csv", inv=False, subset=None, frac=0, prominence=0.2
)
ds3 = parse_auto(
    "./data/ciona/mpx_auto_MAH03787.csv", inv=False, subset=None, frac=0, prominence=0.2
)
ds4 = parse_auto(
    "./data/ciona/mpx_auto_MAH03788.csv", inv=False, subset=None, frac=0, prominence=0.3
)
ds5 = parse_auto(
    "./data/ciona/mpx_auto_MAH03789.csv", inv=False, subset=None, frac=0, prominence=0.3
)
ds6 = parse_auto(
    "./data/ciona/mpx_auto_MAH03790.csv", inv=False, subset=None, frac=0, prominence=0.3
)
ds7 = parse_auto(
    "./data/ciona/mpx_auto_MAH03791.csv", inv=False, subset=None, frac=0, prominence=0.15
)
ds8 = parse_auto(
    "./data/ciona/mpx_auto_MAH03792.csv", inv=False, subset=None, frac=0, prominence=0.15
)
ds9 = parse_auto(
    "./data/ciona/mpx_auto_MAH03793.csv", inv=False, subset=None, frac=0, prominence=0.15
)
ds10 = parse_auto(
    "./data/ciona/mpx_auto_MAH03794.csv", inv=False, subset=None, frac=0, prominence=0.15
)
ds11 = parse_auto(
    "./data/ciona/mpx_auto_MAH03795.csv", inv=False, subset=None, frac=0, prominence=0.15
)
ds12 = parse_auto(
    "./data/ciona/mpx_auto_MAH03795.csv", inv=False, subset=None, frac=0, prominence=0.15
)
ds15 = parse_auto(
    "./data/ciona/mpx_auto_MAH03798.csv", inv=False, subset=None, frac=0, prominence=0.11
)
ds16 = parse_auto(
    "./data/ciona/mpx_auto_MAH03799.csv", inv=False, subset=None, frac=0, prominence=0.11
)

# Man
mds = parse_man('./data/ciona/hr2_man_MAH03784.csv')
mds1 = parse_man('./data/ciona/hr2_man_MAH03785.csv') # s[1:]
mds2 = parse_man("./data/ciona/hr2_man_MAH03786.csv")  # d[:-1], s[1:]
mds3 = parse_man("./data/ciona/hr2_man_MAH03787.csv")
mds4 = parse_man("./data/ciona/hr2_man_MAH03788.csv")
mds5 = parse_man("./data/ciona/hr2_man_MAH03789.csv")  # s[1:]
mds6 = parse_man("./data/ciona/hr2_man_MAH03790.csv")  # s[1:]
mds7 = parse_man("./data/ciona/hr2_man_MAH03791.csv")  # s[1:]
mds8 = parse_man("./data/ciona/hr2_man_MAH03792.csv")  # s[1:]
mds9 = parse_man("./data/ciona/hr2_man_MAH03793.csv")
mds10 = parse_man("./data/ciona/hr2_man_MAH03794.csv")
mds11 = parse_man("./data/ciona/hr2_man_MAH03795.csv")
mds12 = parse_man("./data/ciona/hr2_man_MAH03796.csv")
mds15 = parse_man("./data/ciona/hr2_man_MAH03799.csv")
mds16 = parse_man("./data/ciona/hr2_man_MAH03800.csv")

# Create containers
a_hr2 = []
m_hr2 = []
a_min_t2 = []
m_min_t2 = []
a_max_t2 = []
m_max_t2 = []
a_mean_t2 = []
m_mean_t2 = []
a_med_t2 = []
m_med_t2 = []
a_std_t2 = []
m_std_t2 = []
a_rmss = []
m_rmss = []

all_a = [a_hr2, a_min_t2, a_max_t2, a_mean_t2, a_med_t2, a_std_t2, a_rmss]
all_m = [m_hr2, m_min_t2, m_max_t2, m_mean_t2, m_med_t2, m_std_t2, m_rmss]

# Compute stats (auto)
all_a = append_with(all_a, stats(*ds, 300, 25))
all_a = append_with(all_a, stats(*ds1, 300, 25))
all_a = append_with(all_a, stats(*ds2, 300, 25))
all_a = append_with(all_a, stats(*ds3, 300, 25))
all_a = append_with(all_a, stats(*ds4, 300, 25))
all_a = append_with(all_a, stats(*ds5, 300, 25))
all_a = append_with(all_a, stats(*ds6, 300, 25))
all_a = append_with(all_a, stats(*ds7, 300, 25))
all_a = append_with(all_a, stats(*ds8, 300, 25))
all_a = append_with(all_a, stats(*ds9, 300, 25))
all_a = append_with(all_a, stats(*ds10, 300, 25))
all_a = append_with(all_a, stats(*ds11, 300, 25))
all_a = append_with(all_a, stats(*ds15, 300, 25))
all_a = append_with(all_a, stats(*ds16, 300, 25))

# Compute stats (man)
all_m = append_with(all_m, stats(*mds, 300, 25))
all_m = append_with(all_m, stats(*mds1, 300, 25))
all_m = append_with(all_m, stats(*mds2, 300, 25))
all_m = append_with(all_m, stats(*mds3, 300, 25))
all_m = append_with(all_m, stats(*mds4, 300, 25))
all_m = append_with(all_m, stats(*mds5, 300, 25))
all_m = append_with(all_m, stats(*mds6, 300, 25))
all_m = append_with(all_m, stats(*mds7, 300, 25))
all_m = append_with(all_m, stats(*mds8, 300, 25))
all_m = append_with(all_m, stats(*mds9, 300, 25))
all_m = append_with(all_m, stats(*mds10, 300, 25))
all_m = append_with(all_m, stats(*mds11, 300, 25))
all_m = append_with(all_m, stats(*mds15, 300, 25))
all_m = append_with(all_m, stats(*mds16, 300, 25))

# Deconstruct
[a_hr2, a_min_t2, a_max_t2, a_mean_t2, a_med_t2, a_std_t2, a_rmss2] = all_a
[m_hr2, m_min_t2, m_max_t2, m_mean_t2, m_med_t2, m_std_t2, m_rmss2] = all_m

# And plot
fig, ax = plt.subplots()
ax.scatter(m_hr, a_hr, label='P. serratus')
ax.scatter(m_hr1, a_hr1, label='R. balthica')
ax.scatter(m_hr2, a_hr2, label='C. intestinalis')

a_hr.extend(a_hr1)
a_hr.extend(a_hr2)

m_hr.extend(m_hr1)
m_hr.extend(m_hr2)

r, p = corr_stats(*pearsonr(m_hr, a_hr))

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

min, max = lims

plt.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}')
plt.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}')

ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax.set_aspect("equal")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Manual HR (bpm)")
ax.set_ylabel("HeartCV HR (bpm)")
ax.legend(loc='upper left')
plt.show()

# Beat to beat stats
fig, (
    (
        ax1,
        ax2,
        ax3,
    ),
    (ax4, ax5, ax6),
) = plt.subplots(2, 3)
fig.suptitle("Manual (x-axis) vs HeartCV (y-axis) heart rate variability (seconds)")

ax1.scatter(m_min_t, a_min_t, label='P. serratus')
ax1.scatter(m_min_t1, a_min_t1, label='R. balthica')
ax1.scatter(m_min_t2, a_min_t2, label='C. intestinalis')
lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),
    np.max([ax1.get_xlim(), ax1.get_ylim()]),
]

a_min_t.extend(a_min_t1)
a_min_t.extend(a_min_t2)

m_min_t.extend(m_min_t1)
m_min_t.extend(m_min_t2)

r, p = corr_stats(*pearsonr(m_min_t, a_min_t))

min, max = lims

ax1.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}')
ax1.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}')

ax1.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax1.set_aspect("equal")
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_title("Min")

ax2.scatter(m_max_t, a_max_t, label='P. serratus')
ax2.scatter(m_max_t1, a_max_t1, label='R. balthica')
ax2.scatter(m_max_t2, a_max_t2, label='C. intestinalis')
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),
    np.max([ax2.get_xlim(), ax2.get_ylim()]),
]

a_max_t.extend(a_max_t1)
a_max_t.extend(a_max_t2)

m_max_t.extend(m_max_t1)
m_max_t.extend(m_max_t2)

r, p = corr_stats(*pearsonr(m_max_t, a_max_t))

min, max = lims

ax2.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}')
ax2.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}')

ax2.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax2.set_aspect("equal")
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_title("Max")

ax3.scatter(m_mean_t, a_mean_t, label='P. serratus')
ax3.scatter(m_mean_t1, a_mean_t1, label='R. balthica')
ax3.scatter(m_mean_t2, a_mean_t2, label='C. intestinalis')
lims = [
    np.min([ax3.get_xlim(), ax3.get_ylim()]),
    np.max([ax3.get_xlim(), ax3.get_ylim()]),
]

a_mean_t.extend(a_mean_t1)
a_mean_t.extend(a_mean_t2)

m_mean_t.extend(m_mean_t1)
m_mean_t.extend(m_mean_t2)

r, p = corr_stats(*pearsonr(m_mean_t, a_mean_t))

min, max = lims

ax3.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}')
ax3.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}')

ax3.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax3.set_aspect("equal")
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_title("Mean")

ax4.scatter(m_med_t, a_med_t, label='P. serratus')
ax4.scatter(m_med_t1, a_med_t1, label='R. balthica')
ax4.scatter(m_med_t2, a_med_t2, label='C. intestinalis')
lims = [
    np.min([ax4.get_xlim(), ax4.get_ylim()]),
    np.max([ax4.get_xlim(), ax4.get_ylim()]),
]

a_med_t.extend(a_med_t1)
a_med_t.extend(a_med_t2)

m_med_t.extend(m_med_t1)
m_med_t.extend(m_med_t2)

r, p = corr_stats(*pearsonr(m_med_t, a_med_t))

min, max = lims

ax4.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}')
ax4.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}')

ax4.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax4.set_aspect("equal")
ax4.set_xlim(lims)
ax4.set_ylim(lims)
ax4.set_title("Median")

ax5.scatter(m_std_t, a_std_t, label='P. serratus')
ax5.scatter(m_std_t1, a_std_t1, label='R. balthica')
ax5.scatter(m_std_t2, a_std_t2, label='C. intestinalis')
lims = [
    np.min([ax5.get_xlim(), ax5.get_ylim()]),
    np.max([ax5.get_xlim(), ax5.get_ylim()]),
]

a_std_t.extend(a_std_t1)
a_std_t.extend(a_std_t2)

m_std_t.extend(m_std_t1)
m_std_t.extend(m_std_t2)

r, p = corr_stats(*pearsonr(m_std_t, a_std_t))

min, max = lims

ax5.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}')
ax5.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}')

ax5.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax5.set_aspect("equal")
ax5.set_xlim(lims)
ax5.set_ylim(lims)
ax5.set_title("Std")

ax1.legend(loc='upper left')

plt.show()

