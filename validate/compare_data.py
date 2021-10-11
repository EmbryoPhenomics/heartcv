from heartcv import *
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import signal
import heartcv
import re

def corr_stats(r, p, n):
    if p <= 0.05: # *
        if p <= 0.01: # **
            if p <= 0.001: # ***
                if p <= 0.0001: # ****
                    p_str = 'p < 0.0001'
                else:
                    p_str = 'p < 0.001'
            else:
                p_str = 'p < 0.01'
        else:
            p_str = 'p < 0.05'
    else:
        p_str = 'p > 0.05'

    df = n-2
    r_str = f'$r_{{47}}$ = {round(r, 3)}'

    return r_str, p_str

def parse(file, column, upto):
    dat = pd.read_csv(file)
    mpx = dat[column]
    mpx = mpx[:upto]

    return np.asarray(mpx)

def parse_man(file, column, upto):
    dat = pd.read_csv(file)
    d_peaks = dat[column]
    d_peaks = d_peaks[d_peaks < upto]

    return np.asarray(d_peaks)

man_files = [
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_young_1.csv",  
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_young_A1_37.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_young_A1_60.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_young_A4_60.csv",  
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_young_C12_1.csv",  
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_young_D3_50.csv",  
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_young_H3_90.csv",  
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_medium_B2_60.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_medium_C2_149.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_medium_G2_113.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_medium_G5_1.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_medium_G1_82.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_old_B5_26.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_old_B8_1.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_old_D10_42.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_old_E5_15.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_old_F4_1.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_old_F10_21.csv",
    "/home/z/github/heartcv_main/validation/data/paleomon/hr_man_15_15ppt_old_H6.csv"
]

hcv_files = [
    "./hcv_data/paleomon/15_15ppt_young_1.csv",  
    "./hcv_data/paleomon/15_15ppt_young_A1_37.csv",
    "./hcv_data/paleomon/15_15ppt_young_A1_60.csv",
    "./hcv_data/paleomon/15_15ppt_young_A4_60.csv",  
    "./hcv_data/paleomon/15_15ppt_young_C12_1.csv",  
    "./hcv_data/paleomon/15_15ppt_young_D3_50.csv",  
    "./hcv_data/paleomon/15_15ppt_young_H3_90.csv",  
    "./hcv_data/paleomon/15_15ppt_medium_B2_60.csv",
    "./hcv_data/paleomon/15_15ppt_medium_C2_149.csv",
    "./hcv_data/paleomon/15_15ppt_medium_G2_113.csv",
    "./hcv_data/paleomon/15_15ppt_medium_G5_1.csv",
    "./hcv_data/paleomon/15_15ppt_medium_G1_82.csv",
    "./hcv_data/paleomon/15_15ppt_old_B5_26.csv",
    "./hcv_data/paleomon/15_15ppt_old_B8_1.csv",
    "./hcv_data/paleomon/15_15ppt_old_D10_42.csv",
    "./hcv_data/paleomon/15_15ppt_old_E5_15.csv",
    "./hcv_data/paleomon/15_15ppt_old_F4_1.csv",
    "./hcv_data/paleomon/15_15ppt_old_F10_21.csv",
    "./hcv_data/paleomon/15_15ppt_old_H6.csv"
]

# Analyse
paleomon_hcv_stats = dict(
        bpm=[],
        min_b2b=[],
        mean_b2b=[],
        median_b2b=[],
        max_b2b=[],
        sd_b2b=[],
        range_b2b=[],
        rmssd=[])

paleomon_man_stats = dict(
        bpm=[],
        min_b2b=[],
        mean_b2b=[],
        median_b2b=[],
        max_b2b=[],
        sd_b2b=[],
        range_b2b=[],
        rmssd=[])

for hf, mf in zip(hcv_files, man_files):
    hf_data = np.asarray(parse(hf, 'area', 250))
    hf_data = np.interp([i/3 for i in range(250*3)], np.arange(0, 250), hf_data)
    hf_data = hf_data.max() - hf_data
    hf_peaks = find_peaks(hf_data)
    hf_stats = stats(hf_peaks, 250*3, 25*3)

    mf_peaks = parse_man(mf, 'EndDiastoleFrame', 250)
    mf_stats = stats(mf_peaks, sample_length=250, fs=25)

    for key in paleomon_hcv_stats.keys():
        if key == 'rmssd':
            print(hf)
            print(hf_stats[key], mf_stats[key])
        paleomon_hcv_stats[key].append(hf_stats[key])
        paleomon_man_stats[key].append(mf_stats[key])

# Radix -------------------
man_files = [
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_20deg_A1.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_20deg_A3.csv",  
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_20deg_A4.csv",  
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_20deg_A6.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_20deg_B1.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_20deg_B2.csv",  
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_20deg_B3.csv",  
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_25deg_A1.csv",  
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_25deg_A4.csv",  
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_25deg_A5.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_25deg_A7.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_25deg_B1.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_30deg_B1.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_30deg_B5.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_unknown_C1.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_unknown_C8.csv",
    "/home/z/github/heartcv_main/validation/data/radix/hr_man_unknown_D1.csv"
]

hcv_files = [
    "./hcv_data/radix/20deg_A1.csv",
    "./hcv_data/radix/20deg_A3.csv",  
    "./hcv_data/radix/20deg_A4.csv",
    "./hcv_data/radix/20deg_A6.csv",
    "./hcv_data/radix/20deg_B1.csv",
    "./hcv_data/radix/20deg_B2.csv",  
    "./hcv_data/radix/20deg_B3.csv",
    "./hcv_data/radix/25deg_A1.csv",  
    "./hcv_data/radix/25deg_A4.csv",  
    "./hcv_data/radix/25deg_A5.csv",
    "./hcv_data/radix/25deg_A7.csv",
    "./hcv_data/radix/25deg_B1.csv",
    "./hcv_data/radix/30deg_B1.csv",
    "./hcv_data/radix/30deg_B5.csv",
    "./hcv_data/radix/unknown_C1.csv",
    "./hcv_data/radix/unknown_C8.csv",
    "./hcv_data/radix/unknown_D1.csv"
]

# Analyse
radix_hcv_stats = dict(
        bpm=[],
        min_b2b=[],
        mean_b2b=[],
        median_b2b=[],
        max_b2b=[],
        sd_b2b=[],
        range_b2b=[],
        rmssd=[])

radix_man_stats = dict(
        bpm=[],
        min_b2b=[],
        mean_b2b=[],
        median_b2b=[],
        max_b2b=[],
        sd_b2b=[],
        range_b2b=[],
        rmssd=[])

for hf, mf in zip(hcv_files, man_files):
    hf_data = np.asarray(parse(hf, 'area', 200))
    hf_data = np.interp([i/3 for i in range(200*3)], np.arange(0, 200), hf_data)
    hf_data = hf_data.max() - hf_data
    hf_peaks = find_peaks(hf_data)
    hf_stats = stats(hf_peaks, 200*3, 20*3)

    mf_peaks = parse_man(mf, 'EndDiastoleFrame', 200)
    mf_stats = stats(mf_peaks, sample_length=200, fs=20)

    for key in radix_hcv_stats.keys():
        if key == 'rmssd':
            print(hf)
            print(hf_stats[key], mf_stats[key])
        radix_hcv_stats[key].append(hf_stats[key])
        radix_man_stats[key].append(mf_stats[key])

# Ciona -------------------
man_files = [
    '/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03784.csv',
    '/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03785.csv', 
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03786.csv",
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03787.csv",
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03788.csv",
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03789.csv",  
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03790.csv",  
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03791.csv",  
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03794.csv",
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03795.csv",
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03798.csv",
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03799.csv",
    "/home/z/github/heartcv_main/validation/data/ciona/hr2_man_MAH03800.csv"
]

hcv_files = [
    './hcv_data/ciona/MAH03784.csv',
    './hcv_data/ciona/MAH03785.csv', 
    "./hcv_data/ciona/MAH03786.csv", 
    "./hcv_data/ciona/MAH03787.csv",
    "./hcv_data/ciona/MAH03788.csv",
    "./hcv_data/ciona/MAH03789.csv",
    "./hcv_data/ciona/MAH03790.csv",
    "./hcv_data/ciona/MAH03791.csv",
    "./hcv_data/ciona/MAH03794.csv",
    "./hcv_data/ciona/MAH03795.csv",
    "./hcv_data/ciona/MAH03798.csv",
    "./hcv_data/ciona/MAH03799.csv",
    "./hcv_data/ciona/MAH03800.csv"
]


# Analyse
ciona_hcv_stats = dict(
        bpm=[],
        min_b2b=[],
        mean_b2b=[],
        median_b2b=[],
        max_b2b=[],
        sd_b2b=[],
        range_b2b=[],
        rmssd=[])

ciona_man_stats = dict(
        bpm=[],
        min_b2b=[],
        mean_b2b=[],
        median_b2b=[],
        max_b2b=[],
        sd_b2b=[],
        range_b2b=[],
        rmssd=[])

for hf, mf in zip(hcv_files, man_files):

    hf_data = np.asarray(parse(hf, 'area', 250))
    hf_data = np.interp([i/3 for i in range(250*3)], np.arange(0, 250), hf_data)
    hf_data = hf_data.max() - hf_data
    hf_peaks = find_peaks(hf_data)
    hf_stats = stats(hf_peaks, 250*3, 25*3)

    mf_peaks = parse_man(mf, 'EndDiastoleFrame', 250)
    mf_stats = stats(mf_peaks, sample_length=250, fs=25)

    for key in ciona_hcv_stats.keys():
        ciona_hcv_stats[key].append(hf_stats[key])
        ciona_man_stats[key].append(mf_stats[key])

# Plot
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax1.text(-0.1, 1.1, "a)", transform=ax1.transAxes, size=12)
ax1.set_title('BPM', fontsize=10)
ax1.scatter(paleomon_man_stats['bpm'], paleomon_hcv_stats['bpm'], label='$C. intestinalis$')
ax1.scatter(radix_man_stats['bpm'], radix_hcv_stats['bpm'], label='$R. balthica$')
ax1.scatter(ciona_man_stats['bpm'], ciona_hcv_stats['bpm'], label='$P. serratus$')
lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),
    np.max([ax1.get_xlim(), ax1.get_ylim()]),
]
ax1.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax1.set_aspect("equal")
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.legend(loc='upper left')

m_bpm = paleomon_man_stats['bpm'] + radix_man_stats['bpm'] + ciona_man_stats['bpm']
h_bpm = paleomon_hcv_stats['bpm'] + radix_hcv_stats['bpm'] + ciona_hcv_stats['bpm']
print(len(m_bpm))

r, p = corr_stats(*pearsonr(m_bpm, h_bpm), len(m_bpm))
min, max = lims
ax1.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=6)
ax1.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=6)
ax1.set_xlabel('Manual heart rate (bpm)', fontsize=6)
ax1.set_ylabel('HeartCV heart rate (bpm)', fontsize=6)

ax2.text(-0.1, 1.1, "b)", transform=ax2.transAxes, size=12)
ax2.set_title('Minimum interbeat interval', fontsize=10)
ax2.scatter(paleomon_man_stats['min_b2b'], paleomon_hcv_stats['min_b2b'])
ax2.scatter(radix_man_stats['min_b2b'], radix_hcv_stats['min_b2b'])
ax2.scatter(ciona_man_stats['min_b2b'], ciona_hcv_stats['min_b2b'])
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),
    np.max([ax2.get_xlim(), ax2.get_ylim()]),
]
ax2.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax2.set_aspect("equal")
ax2.set_xlim(lims)
ax2.set_ylim(lims)
m_min_b2b = paleomon_man_stats['min_b2b'] + radix_man_stats['min_b2b'] + ciona_man_stats['min_b2b']
h_min_b2b = paleomon_hcv_stats['min_b2b'] + radix_hcv_stats['min_b2b'] + ciona_hcv_stats['min_b2b']

r, p = corr_stats(*pearsonr(m_min_b2b, h_min_b2b), len(m_min_b2b))
min, max = lims
ax2.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=6)
ax2.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=6)
ax2.set_xlabel('Manual minimum IBI (seconds)', fontsize=6)
ax2.set_ylabel('HeartCV minimum IBI (seconds)', fontsize=6)

ax3.text(-0.1, 1.1, "c)", transform=ax3.transAxes, size=12)
ax3.set_title('Mean interbeat interval', fontsize=10)
ax3.scatter(paleomon_man_stats['mean_b2b'], paleomon_hcv_stats['mean_b2b'])
ax3.scatter(radix_man_stats['mean_b2b'], radix_hcv_stats['mean_b2b'])
ax3.scatter(ciona_man_stats['mean_b2b'], ciona_hcv_stats['mean_b2b'])
lims = [
    np.min([ax3.get_xlim(), ax3.get_ylim()]),
    np.max([ax3.get_xlim(), ax3.get_ylim()]),
]
ax3.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax3.set_aspect("equal")
ax3.set_xlim(lims)
ax3.set_ylim(lims)
m_mean_b2b = paleomon_man_stats['mean_b2b'] + radix_man_stats['mean_b2b'] + ciona_man_stats['mean_b2b']
h_mean_b2b = paleomon_hcv_stats['mean_b2b'] + radix_hcv_stats['mean_b2b'] + ciona_hcv_stats['mean_b2b']

r, p = corr_stats(*pearsonr(m_mean_b2b, h_mean_b2b), len(m_mean_b2b))
min, max = lims
ax3.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=6)
ax3.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=6)
ax3.set_xlabel('Manual mean IBI (seconds)', fontsize=6)
ax3.set_ylabel('HeartCV mean IBI (seconds)', fontsize=6)

ax4.text(-0.1, 1.1, "d)", transform=ax4.transAxes, size=12)
ax4.set_title('Maximum interbeat interval', fontsize=10)
ax4.scatter(paleomon_man_stats['max_b2b'], paleomon_hcv_stats['max_b2b'])
ax4.scatter(radix_man_stats['max_b2b'], radix_hcv_stats['max_b2b'])
ax4.scatter(ciona_man_stats['max_b2b'], ciona_hcv_stats['max_b2b'])
lims = [
    np.min([ax4.get_xlim(), ax4.get_ylim()]),
    np.max([ax4.get_xlim(), ax4.get_ylim()]),
]
ax4.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax4.set_aspect("equal")
ax4.set_xlim(lims)
ax4.set_ylim(lims)
m_max_b2b = paleomon_man_stats['max_b2b'] + radix_man_stats['max_b2b'] + ciona_man_stats['max_b2b']
h_max_b2b = paleomon_hcv_stats['max_b2b'] + radix_hcv_stats['max_b2b'] + ciona_hcv_stats['max_b2b']

r, p = corr_stats(*pearsonr(m_max_b2b, h_max_b2b), len(m_max_b2b))
min, max = lims
ax4.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=6)
ax4.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=6)
ax4.set_xlabel('Manual maximum IBI (seconds)', fontsize=6)
ax4.set_ylabel('HeartCV maximum IBI (seconds)', fontsize=6)

ax5.text(-0.1, 1.1, "e)", transform=ax5.transAxes, size=12)
ax5.set_title('σ in interbeat interval', fontsize=10)
ax5.scatter(paleomon_man_stats['sd_b2b'], paleomon_hcv_stats['sd_b2b'])
ax5.scatter(radix_man_stats['sd_b2b'], radix_hcv_stats['sd_b2b'])
ax5.scatter(ciona_man_stats['sd_b2b'], ciona_hcv_stats['sd_b2b'])
lims = [
    np.min([ax5.get_xlim(), ax5.get_ylim()]),
    np.max([ax5.get_xlim(), ax5.get_ylim()]),
]
ax5.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax5.set_aspect("equal")
ax5.set_xlim(lims)
ax5.set_ylim(lims)
m_sd_b2b = paleomon_man_stats['sd_b2b'] + radix_man_stats['sd_b2b'] + ciona_man_stats['sd_b2b']
h_sd_b2b = paleomon_hcv_stats['sd_b2b'] + radix_hcv_stats['sd_b2b'] + ciona_hcv_stats['sd_b2b']

r, p = corr_stats(*pearsonr(m_sd_b2b, h_sd_b2b), len(m_sd_b2b))
min, max = lims
ax5.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=6)
ax5.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=6)
ax5.set_xlabel('Manual σ IBI (seconds)', fontsize=6)
ax5.set_ylabel('HeartCV σ IBI (seconds)', fontsize=6)

ax6.text(-0.1, 1.1, "f)", transform=ax6.transAxes, size=12)
ax6.set_title('RMSSD', fontsize=10)
ax6.scatter(paleomon_man_stats['rmssd'], paleomon_hcv_stats['rmssd'])
ax6.scatter(radix_man_stats['rmssd'], radix_hcv_stats['rmssd'])
ax6.scatter(ciona_man_stats['rmssd'], ciona_hcv_stats['rmssd'])
lims = [
    np.min([ax6.get_xlim(), ax6.get_ylim()]),
    np.max([ax6.get_xlim(), ax6.get_ylim()]),
]
ax6.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax6.set_aspect("equal")
ax6.set_xlim(lims)
ax6.set_ylim(lims)
m_rmssd = paleomon_man_stats['rmssd'] + radix_man_stats['rmssd'] + ciona_man_stats['rmssd']
h_rmssd = paleomon_hcv_stats['rmssd'] + radix_hcv_stats['rmssd'] + ciona_hcv_stats['rmssd']

r, p = corr_stats(*pearsonr(m_rmssd, h_rmssd), len(m_rmssd))

min, max = lims
ax6.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=6)
ax6.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=6)
ax6.set_xlabel('Manual RMSSD (seconds)', fontsize=6)
ax6.set_ylabel('HeartCV RMSSD (seconds)', fontsize=6)

plt.show()