import heartcv as hcv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from heartcv.core.segmentation import _minmax_scale as scale

def plot(man, auto):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(man, label='Manual')
    ax1.plot(auto, label='Auto')
    ax1.legend(loc='lower right')
    ax1.set_ylabel('Area')
    ax1.set_xlabel('Frame')

    ax2.scatter(man, auto)
    ax2.set_ylabel('Auto')
    ax2.set_xlabel('Manual')

    plt.show()

def plot_events(man, auto, events_man, events_auto):

    man_d, man_s = events_man[1:]
    auto_d, auto_s = events_auto[1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(man)
    ax1.plot(*man_d, 'x')
    ax1.plot(*man_s, 'x')
    ax1.set_ylabel('Area')

    ax2.plot(auto)
    ax2.plot(*auto_d, 'x')
    ax2.plot(*auto_s, 'x')
    ax2.set_ylabel('Area')
    ax2.set_xlabel('Frame')

    plt.show()

def parse(man, auto, up_to=100, every=2, left=0, inv_auto=False, rolling=False, win_size=3):
    man = pd.read_csv(man)[:up_to]
    auto = pd.read_csv(auto)[:up_to]

    man_areas = man['area'][:up_to]
    auto_areas = auto['area'][:up_to]

    man_areas = [n for i,n in enumerate(man_areas) if i%every == left]
    auto_areas = [n for i,n in enumerate(auto_areas) if i%every == left]

    if inv_auto:
        auto_areas = max(auto_areas) - np.asarray(auto_areas)

    if rolling:
        # man_df = pd.DataFrame(data=dict(x=man_areas))
        auto_df = pd.DataFrame(data=dict(x=auto_areas))
        # man_areas = man_df['x'].rolling(win_size).mean()
        auto_areas = auto_df['x'].rolling(win_size).mean()

    man_areas, auto_areas = map(scale, (man_areas, auto_areas))

    return (man_areas.tolist(), auto_areas.tolist())        

def mse(truth, act):
    truth, act = map(np.asarray, (truth, act))
    truth, act = map(scale, (truth, act))
    diff = truth - act
    return np.nanmean(diff**2)

def rmse(truth, act):
    return np.sqrt(mse(truth, act))

def sv(dia, sys):
    dia,sys = map(np.asarray, (dia,sys))
    sv = dia - sys
    return sv

def stats(man_d, man_df, man_s, man_sf, auto_d, auto_df, auto_s, auto_sf):
    mansv = sv(man_d, man_s)
    autosv = sv(auto_d, auto_s)

    mansv,autosv = map(scale, (mansv, autosv))

    svmse = mse(mansv, autosv)
    dmse = mse(man_d, auto_d)
    smse = mse(man_s, auto_s)

    svrmse = rmse(mansv, autosv)
    drmse = rmse(man_d, auto_d)
    srmse = rmse(man_s, auto_s)

    svsv = pearsonr(mansv, autosv)
    dd = pearsonr(man_d, auto_d)
    ss = pearsonr(auto_d, auto_d)

    doffset = [m - a for m,a in zip(man_df, auto_df)]
    soffset = [m - a for m,a in zip(man_sf, auto_sf)]

    return (mansv, autosv, svmse, dmse, smse, svrmse, drmse, srmse, svsv, dd, ss, doffset, soffset)

def area_stats(man, auto):
    amse = mse(man,auto)
    armse = rmse(man,auto)
    aa = pearsonr(man,auto)
    return (amse,armse,aa)

def scatter(x, y, xlabel, ylabel, path):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.show()

# Paleomon

man_areas = []
auto_areas = []
man_d = []
man_s = []
auto_d = []
auto_s = []

man_df = []
man_sf = []
auto_df = []
auto_sf = []

man, auto = parse(
    './data/paleomon/sv_man_15_15ppt_young_1(1).csv', 
    './data/paleomon/sv_auto_15_15ppt_young_1.csv',
    100, 1, 0, False)
events_man = hcv.find_events(man, prominence=0.3)
events_auto = hcv.find_events(auto, prominence=0.2)
# plot(man, auto)

man_areas = man_areas + man
auto_areas = auto_areas + auto

# plot_events(man, auto, events_man, events_auto)

man_d = man_d + events_man[1][1][1:].tolist()
man_s = man_s + events_man[2][1][1:].tolist()
auto_d = auto_d + events_auto[1][1].tolist()
auto_s = auto_s + events_auto[2][1][1:].tolist()

man_df = man_df + events_man[1][0][1:].tolist()
man_sf = man_sf + events_man[2][0][1:].tolist()
auto_df = auto_df + events_auto[1][0].tolist()
auto_sf = auto_sf + events_auto[2][0][1:].tolist()

man, auto = parse(
    './data/paleomon/sv_man_15_15ppt_young_C12_1.csv', 
    './data/paleomon/sv_auto_15_15ppt_young_C12_1.csv',
    100, 2, 0, True)
events_man = hcv.find_events(man, prominence=0.3)
events_auto = hcv.find_events(auto, prominence=0.2)
plot(man, auto)

man_areas = man_areas + man
auto_areas = auto_areas + auto

plot_events(man, auto, events_man, events_auto)

man_d = man_d + events_man[1][1].tolist()
man_s = man_s + events_man[2][1][1:].tolist()
auto_d = auto_d + events_auto[1][1].tolist()
auto_s = auto_s + events_auto[2][1][1:].tolist()

man_df = man_df + events_man[1][0].tolist()
man_sf = man_sf + events_man[2][0][1:].tolist()
auto_df = auto_df + events_auto[1][0].tolist()
auto_sf = auto_sf + events_auto[2][0][1:].tolist()

man, auto = parse(
  './data/paleomon/sv_man_15_15ppt_medium_1.csv', 
  './data/paleomon/sv_auto_15_15ppt_medium_1.csv',
  100, 2, 1, True)
# plot(man, auto)
man_areas = man_areas + man
auto_areas = auto_areas + auto

man, auto = parse(
  './data/paleomon/sv_man_15_15ppt_old_1.csv', 
  './data/paleomon/sv_auto_15_15ppt_old_1.csv',
  100, 2, 0, False)
# plot(man, auto)

man_areas = man_areas + man
auto_areas = auto_areas + auto

man, auto = parse(
  './data/paleomon/sv_manz_15_15ppt_young_A1_37.csv', 
  './data/paleomon/sv_auto_15_15ppt_young_A1_37.csv',
  53, 1, 0, True)
events_man = hcv.find_events(man, prominence=0.3)
events_auto = hcv.find_events(auto, prominence=0.2)
# plot(man, auto)

man_areas = man_areas + man
auto_areas = auto_areas + auto

# plot_events(man, auto, events_man, events_auto)

man_d = man_d + events_man[1][1].tolist()
man_s = man_s + events_man[2][1][1:].tolist()
auto_d = auto_d + events_auto[1][1][1:].tolist()
auto_s = auto_s + events_auto[2][1][1:].tolist()

man_df = man_df + events_man[1][0].tolist()
man_sf = man_sf + events_man[2][0][1:].tolist()
auto_df = auto_df + events_auto[1][0][1:].tolist()
auto_sf = auto_sf + events_auto[2][0][1:].tolist()

man, auto = parse(
  './data/paleomon/sv_man_15_15ppt_young_A4_60.csv', 
  './data/paleomon/sv_auto_15_15ppt_young_A4_60.csv',
  100, 2, 0, False)
events_man = hcv.find_events(man, prominence=0.3)
events_auto = hcv.find_events(auto, prominence=0.2)
# plot(man, auto)

man_areas = man_areas + man
auto_areas = auto_areas + auto

# plot_events(man, auto, events_man, events_auto)

man_d = man_d + events_man[1][1].tolist()
man_s = man_s + events_man[2][1][1:].tolist()
auto_d = auto_d + events_auto[1][1].tolist()
auto_s = auto_s + events_auto[2][1][1:].tolist()

man_df = man_df + events_man[1][0].tolist()
man_sf = man_sf + events_man[2][0][1:].tolist()
auto_df = auto_df + events_auto[1][0].tolist()
auto_sf = auto_sf + events_auto[2][0][1:].tolist()

man, auto = parse(
  './data/paleomon/sv_man_15_15ppt_young_A4_60.csv', 
  './data/paleomon/sv_auto_15_15ppt_young_A4_60.csv',
  100, 2, 0, False)
events_man = hcv.find_events(man, prominence=0.3)
events_auto = hcv.find_events(auto, prominence=0.2)
# plot(man, auto)

man_areas = man_areas + man
auto_areas = auto_areas + auto

# plot_events(man, auto, events_man, events_auto)

man_d = man_d + events_man[1][1].tolist()
man_s = man_s + events_man[2][1][1:].tolist()
auto_d = auto_d + events_auto[1][1].tolist()
auto_s = auto_s + events_auto[2][1][1:].tolist()

man_df = man_df + events_man[1][0].tolist()
man_sf = man_sf + events_man[2][0][1:].tolist()
auto_df = auto_df + events_auto[1][0].tolist()
auto_sf = auto_sf + events_auto[2][0][1:].tolist()

# plt.scatter(man_areas, auto_areas)
# plt.xlabel('Manual')
# plt.ylabel('HeartCV')
# plt.show()

(mansv, autosv, svmse, dmse, smse, svrmse, drmse, srmse, svsv, dd, ss, doffset, soffset) = stats(man_d, man_df, man_s, man_sf, auto_d, auto_df, auto_s, auto_sf)
(amse,armse,aa) = area_stats(man_areas, auto_areas)

scatter(man_areas, auto_areas, 'Manual', 'Heartcv', './plots/paleomon_areas.png')
scatter(mansv, autosv, 'Manual', 'Heartcv', './plots/paleomon_sv.png')
scatter(man_d, auto_d, 'Manual', 'Heartcv', './plots/paleomon_d.png')
scatter(man_s, auto_s, 'Manual', 'Heartcv', './plots/paleomon_s.png')

# Radix

man_areasr = []
auto_areasr = []

man_dr = []
man_sr = []
auto_dr = []
auto_sr = []

man_dfr = []
man_sfr = []
auto_dfr = []
auto_sfr = []

man, auto = parse(
  './data/radix/sv_auto_20deg_A1.csv', 
  './data/radix/sv_man_20deg_A1.csv',
  100, 2, 0, True)
events_man = hcv.find_events(man, prominence=0.2)
events_auto = hcv.find_events(auto, prominence=0.2)

man_areasr = man_areasr + man
auto_areasr = auto_areasr + auto
# plot_events(man, auto, events_man, events_auto)

man_dr = man_dr + events_man[1][1][1:].tolist()
man_sr = man_sr + events_man[2][1][1:].tolist()
auto_dr = auto_dr + events_auto[1][1].tolist()
auto_sr = auto_sr + events_auto[2][1][1:].tolist()

man_dfr = man_dfr + events_man[1][0][1:].tolist()
man_sfr = man_sfr + events_man[2][0][1:].tolist()
auto_dfr = auto_dfr + events_auto[1][0].tolist()
auto_sfr = auto_sfr + events_auto[2][0][1:].tolist()


man, auto = parse(
  './data/radix/sv_auto_20deg_A3.csv', 
  './data/radix/sv_man_20deg_A3.csv',
  100, 2, 0, False)
events_man = hcv.find_events(man, prominence=0.2)
events_auto = hcv.find_events(auto, prominence=0.2)

man_areasr = man_areasr + man
auto_areasr = auto_areasr + auto
# plot_events(man, auto, events_man, events_auto)

man_dr = man_dr + events_man[1][1][:-1].tolist()
man_sr = man_sr + events_man[2][1].tolist()
auto_dr = auto_dr + events_auto[1][1][:-1].tolist()
auto_sr = auto_sr + events_auto[2][1].tolist()

man_dfr = man_dfr + events_man[1][0][:-1].tolist()
man_sfr = man_sfr + events_man[2][0].tolist()
auto_dfr = auto_dfr + events_auto[1][0][:-1].tolist()
auto_sfr = auto_sfr + events_auto[2][0].tolist()

man, auto = parse(
  './data/radix/sv_auto_20deg_A4.csv', 
  './data/radix/sv_man_20deg_A4.csv',
  100, 2, 0, True)
events_man = hcv.find_events(man, prominence=0.2)
events_auto = hcv.find_events(auto, prominence=0.2)

man_areasr = man_areasr + man
auto_areasr = auto_areasr + auto
# plot_events(man, auto, events_man, events_auto)

man_dr = man_dr + events_man[1][1].tolist()
man_sr = man_sr + events_man[2][1].tolist()
auto_dr = auto_dr + events_auto[1][1].tolist()
auto_sr = auto_sr + events_auto[2][1].tolist()

man_dfr = man_dfr + events_man[1][0].tolist()
man_sfr = man_sfr + events_man[2][0].tolist()
auto_dfr = auto_dfr + events_auto[1][0].tolist()
auto_sfr = auto_sfr + events_auto[2][0].tolist()

man, auto = parse(
  './data/radix/sv_auto_20deg_A6.csv', 
  './data/radix/sv_man_20deg_A6.csv',
  100, 2, 0, False)
events_man = hcv.find_events(man, prominence=0.15)
events_auto = hcv.find_events(auto, prominence=0.15)

man_areasr = man_areasr + man
auto_areasr = auto_areasr + auto
# plot_events(man, auto, events_man, events_auto)

man_dr = man_dr + events_man[1][1][:-1].tolist()
man_sr = man_sr + events_man[2][1].tolist()
auto_dr = auto_dr + events_auto[1][1][:-1].tolist()
auto_sr = auto_sr + events_auto[2][1].tolist()

man_dfr = man_dfr + events_man[1][0][:-1].tolist()
man_sfr = man_sfr + events_man[2][0].tolist()
auto_dfr = auto_dfr + events_auto[1][0][:-1].tolist()
auto_sfr = auto_sfr + events_auto[2][0].tolist()

(mansvr, autosvr, svmser, dmser, smser, svrmser, drmser, srmser, svsvr, ddr, ssr, doffsetr, soffsetr) = stats(man_dr, man_dfr, man_sr, man_sfr, auto_dr, auto_dfr, auto_sr, auto_sfr)
(amser,armser,aar) = area_stats(man_areasr, auto_areasr)

scatter(man_areasr, auto_areasr, 'Manual', 'Heartcv', './plots/radix_areas.png')
scatter(mansvr, autosvr, 'Manual', 'Heartcv', './plots/radix_sv.png')
scatter(man_dr, auto_dr, 'Manual', 'Heartcv', './plots/radix_d.png')
scatter(man_sr, auto_sr, 'Manual', 'Heartcv', './plots/radix_s.png')

with open('./mean_px_results.md', 'w') as md:
    text = f'''
# Heart validation results (mean px)

## Summary 

Note all comparisons are manual ~ automated.

### Paleomon 

Comparison type | Pearson r | Pearson p | MSE | RMSE | Offset mean | Offset sd
--- | --- | --- | --- | --- | --- | ---
Raw area (Fig. 1) | {aa[0]} | {aa[1]} | {amse} | {armse} |  |  
Stroke volume (Fig. 2) | {svsv[0]} | {svsv[1]} | {svmse} | {svrmse} |  | 
Diastole | {dd[0]} | {dd[1]} | {dmse} | {drmse} | {np.nanmean(doffset)} | {np.std(doffset)}
Systole | {ss[0]} | {ss[1]} | {smse} | {srmse} | {np.nanmean(soffset)} | {np.std(soffset)}

### Radix 

Comparison type | Pearson r | Pearson p | MSE | RMSE | Offset mean | Offset sd
--- | --- | --- | --- | --- | --- | ---
Raw area (Fig. 3) | {aar[0]} | {aar[1]} | {amser} | {armser} |  | 
Stroke volume (Fig. 4) | {svsvr[0]} | {svsvr[1]} | {svmser} | {svrmser} |  | 
Diastole | {ddr[0]} | {ddr[1]} | {dmser} | {drmser} | {np.nanmean(doffsetr)} | {np.std(doffsetr)}
Systole | {ssr[0]} | {ssr[1]} | {smser} | {srmser} | {np.nanmean(soffsetr)} | {np.std(soffsetr)}

## Comparison plots - Paleomon

### Raw area 

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/paleomon_areas.png'>

**Figure 1.** Validation results for manual and automated raw area measures.

### Stroke volume

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/paleomon_sv.png'>

**Figure 2.** Validation results for manual and automated stroke volume measures.

### Diastole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/paleomon_d.png'>

**Figure 3.** Validation results for manual and automated diastole measures.

### Systole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/paleomon_s.png'>

**Figure 4.** Validation results for manual and automated systole measures.

## Comparison plots - radix

### Raw area 

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/radix_areas.png'>

**Figure 5.** Validation results for manual and automated raw area measures.

### Stroke volume

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/radix_sv.png'>

**Figure 6.** Validation results for manual and automated stroke volume measures.

### Diastole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/radix_d.png'>

**Figure 7.** Validation results for manual and automated diastole measures.

### Systole

<img src='https://github.com/zibbini/misc_embryoPhenomics/blob/master/python/stroke-volume/heartcv/validation/plots/radix_s.png'>

**Figure 8.** Validation results for manual and automated systole measures.

    '''

    md.write(text)

