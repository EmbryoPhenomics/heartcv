
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

    n_ = n-2    
    r_str = f'$r_{{65}}$ = {round(r, 3)}' # 65 is number of samples - 2.

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
    './hcv_data/paleomon/manual_young_C3.csv',
    './hcv_data/paleomon/manual_old_G4.csv',
    './hcv_data/paleomon/manual_medium_C10.csv',
    './hcv_data/paleomon/manual_old_D7.csv',
    './hcv_data/paleomon/manual_old_H2.csv',
    './hcv_data/paleomon/manual_medium_E4.csv',
    './hcv_data/paleomon/manual_old_G10.csv',
    './hcv_data/paleomon/manual_young_A8.csv',
    './hcv_data/paleomon/manual_medium_E1.csv',
    './hcv_data/paleomon/manual_medium_A3.csv',
    './hcv_data/paleomon/manual_medium_F2.csv',
    './hcv_data/paleomon/manual_medium_H5.csv',
    './hcv_data/paleomon/manual_young_E6.csv',
    './hcv_data/paleomon/manual_young_B7.csv',
    './hcv_data/paleomon/manual_old_B1.csv',
    './hcv_data/paleomon/manual_medium_A6.csv',
    './hcv_data/paleomon/manual_old_D8.csv',
    './hcv_data/paleomon/manual_old_B11.csv',
    './hcv_data/paleomon/manual_young_H8.csv',
    './hcv_data/paleomon/manual_young_D5.csv',
    './hcv_data/paleomon/manual_young_G7.csv',
    './hcv_data/paleomon/manual_old_B4.csv',
    './hcv_data/paleomon/manual_young_A4.csv',
    './hcv_data/paleomon/manual_medium_F11.csv',
]

hcv_files = [
    './hcv_data/paleomon/young_C3.csv', 
    './hcv_data/paleomon/old_G4.csv', 
    './hcv_data/paleomon/medium_C10.csv',
    './hcv_data/paleomon/old_D7.csv',
    './hcv_data/paleomon/old_H2.csv',
    './hcv_data/paleomon/medium_E4.csv',
    './hcv_data/paleomon/old_G10.csv',
    './hcv_data/paleomon/young_A8.csv',
    './hcv_data/paleomon/medium_E1.csv',
    './hcv_data/paleomon/medium_A3.csv',
    './hcv_data/paleomon/medium_F2.csv',
    './hcv_data/paleomon/medium_H5.csv',
    './hcv_data/paleomon/young_E6.csv',
    './hcv_data/paleomon/young_B7.csv',
    './hcv_data/paleomon/old_B1.csv',
    './hcv_data/paleomon/medium_A6.csv',
    './hcv_data/paleomon/old_D8.csv',
    './hcv_data/paleomon/old_B11.csv',
    './hcv_data/paleomon/young_H8.csv',
    './hcv_data/paleomon/young_D5.csv',
    './hcv_data/paleomon/young_G7.csv',
    './hcv_data/paleomon/old_B4.csv',
    './hcv_data/paleomon/young_A4.csv',
    './hcv_data/paleomon/medium_F11.csv',
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

    hf_df = pd.read_csv(hf)
    hf_stats = {}
    for key in paleomon_hcv_stats.keys():
        hf_stats[key] = hf_df[key][0]

    try:
        mf_peaks = parse_man(mf, 'end_diastole_frame', 300)
    except:
        mf_peaks = parse_man(mf, 'EndDiastoleFrame', 300)

    mf_stats = stats(mf_peaks, sample_length=300, fs=25)

    for key in paleomon_hcv_stats.keys():
        if key == 'rmssd':
            print(hf)
            print(hf_stats[key], mf_stats[key])
        paleomon_hcv_stats[key].append(hf_stats[key])
        paleomon_man_stats[key].append(mf_stats[key])

# Radix -------------------
man_files = [
    './hcv_data/radix/20C_manual_20C__A1.csv',
    './hcv_data/radix/20C_manual_20C__A2_10d.csv',
    './hcv_data/radix/20C_manual_20C__B6_124801.csv',
    './hcv_data/radix/20C_manual_20C__E3_10d.csv',
    './hcv_data/radix/20C_manual_20C__F1_10d.csv',
    './hcv_data/radix/20C_manual_D6_hippo_247_out.csv',
    './hcv_data/radix/20C_manual_E2_hippo_225_out.csv',
    './hcv_data/radix/20C_manual_E8_hippo_231_out.csv',
    './hcv_data/radix/20C_manual_F5_hippo_241_out.csv',
    './hcv_data/radix/20C_manual_F7_hippo_243_out.csv',
    './hcv_data/radix/25C_manual_25C_A5_7d.csv',
    './hcv_data/radix/25C_manual_25C_B6_6d.csv',
    './hcv_data/radix/25C_manual_25C_D2_80401.csv',
    './hcv_data/radix/25C_manual_25C_D5_7d.csv',
    './hcv_data/radix/25C_manual_25C_E8_7d.csv',
    './hcv_data/radix/25C_manual_C3_hippo_166_out.csv',
    './hcv_data/radix/25C_manual_C8_hippo_163_out.csv',
    './hcv_data/radix/25C_manual_D5_hippo_167_out.csv',
    './hcv_data/radix/25C_manual_E7_hippo_173_out.csv',
    './hcv_data/radix/25C_manual_F4_hippo_164_out.csv',
    './hcv_data/radix/30C_manual_30C_A6_6d.csv',
    './hcv_data/radix/30C_manual_30C_B1_6d.csv',
    './hcv_data/radix/30C_manual_30C_D8_6d.csv',
    './hcv_data/radix/30C_manual_30C_E1_55201.csv',
    './hcv_data/radix/30C_manual_D1_hippo_173_out.csv',
    './hcv_data/radix/30C_manual_D7_hippo_135_out.csv',
    './hcv_data/radix/30C_manual_E6_hippo_163_out.csv',
    './hcv_data/radix/30C_manual_F7_hippo_153_out.csv',
]

hcv_files = [
    './hcv_data/radix/20C_A1.csv',
    './hcv_data/radix/20C_A2_10d.csv',
    './hcv_data/radix/20C_B6_124801.csv',
    './hcv_data/radix/20C_E3_10d.csv',
    './hcv_data/radix/20C_F1_10d.csv',
    './hcv_data/radix/20C_D6_hippo_247_out.csv',
    './hcv_data/radix/20C_E2_hippo_225_out.csv',
    './hcv_data/radix/20C_E8_hippo_231_out.csv',
    './hcv_data/radix/20C_F5_hippo_241_out.csv',
    './hcv_data/radix/20C_F7_hippo_243_out.csv',
    './hcv_data/radix/25C_A5_7d.csv',
    './hcv_data/radix/25C_B6_6d.csv',
    './hcv_data/radix/25C_D2_80401.csv',
    './hcv_data/radix/25C_D5_7d.csv',
    './hcv_data/radix/25C_E8_7d.csv',
    './hcv_data/radix/25C_C3_hippo_166_out.csv',
    './hcv_data/radix/25C_C8_hippo_163_out.csv',
    './hcv_data/radix/25C_D5_hippo_167_out.csv',
    './hcv_data/radix/25C_E7_hippo_173_out.csv',
    './hcv_data/radix/25C_F4_hippo_164_out.csv',
    './hcv_data/radix/30C_A6_6d.csv',
    './hcv_data/radix/30C_B1_6d.csv',
    './hcv_data/radix/30C_D8_6d.csv',
    './hcv_data/radix/30C_E1_55201.csv',
    './hcv_data/radix/30C_D1_hippo_173_out.csv',
    './hcv_data/radix/30C_D7_hippo_135_out.csv',
    './hcv_data/radix/30C_E6_hippo_163_out.csv',
    './hcv_data/radix/30C_F7_hippo_153_out.csv',
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

    hf_df = pd.read_csv(hf)
    hf_stats = {}
    for key in radix_hcv_stats.keys():
        hf_stats[key] = hf_df[key][0]

    try:
        mf_peaks = parse_man(mf, 'end_diastole_frame', 300)
    except:
        mf_peaks = parse_man(mf, 'EndDiastoleFrame', 300)
    mf_stats = stats(mf_peaks, sample_length=300, fs=20)

    for key in radix_hcv_stats.keys():
        if key == 'rmssd':
            print(hf)
            print(hf_stats[key], mf_stats[key])
        radix_hcv_stats[key].append(hf_stats[key])
        radix_man_stats[key].append(mf_stats[key])

# Ciona -------------------
man_files = [
    './hcv_data/ciona/hr2_man_MAH03784.csv',
    './hcv_data/ciona/hr2_man_MAH03785.csv', 
    "./hcv_data/ciona/hr2_man_MAH03786.csv",
    "./hcv_data/ciona/hr2_man_MAH03787.csv",
    "./hcv_data/ciona/hr2_man_MAH03788.csv",
    "./hcv_data/ciona/hr2_man_MAH03789.csv",  
    "./hcv_data/ciona/hr2_man_MAH03790.csv",  
    "./hcv_data/ciona/hr2_man_MAH03791.csv",  
    "./hcv_data/ciona/hr2_man_MAH03794.csv",
    "./hcv_data/ciona/hr2_man_MAH03795.csv",
    "./hcv_data/ciona/hr2_man_MAH03798.csv",
    "./hcv_data/ciona/hr2_man_MAH03799.csv",
    "./hcv_data/ciona/hr2_man_MAH03800.csv"
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

    hf_df = pd.read_csv(hf)
    hf_stats = {}
    for key in radix_hcv_stats.keys():
        hf_stats[key] = hf_df[key][0]

    mf_peaks = parse_man(mf, 'EndDiastoleFrame', 250)
    mf_stats = stats(mf_peaks, sample_length=250, fs=25)

    for key in ciona_hcv_stats.keys():
        if key == 'rmssd':
            print(hf)
            print(hf_stats[key], mf_stats[key])
        ciona_hcv_stats[key].append(hf_stats[key])
        ciona_man_stats[key].append(mf_stats[key])


# df_man = pd.DataFrame(data=ciona_man_stats)
# df_hcv = pd.DataFrame(data=ciona_hcv_stats)

# df_man.to_csv(f'{out_dir}/{result}_manual.csv')
# df_hcv.to_csv(f'{out_dir}/{result}_hcv.csv')

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax1.text(-0.1, 1.1, "A", transform=ax1.transAxes, size=14, weight='bold')
ax1.set_title('BPM', fontsize=10)
ax1.scatter(paleomon_man_stats['bpm'], paleomon_hcv_stats['bpm'], label='$P. serratus$')
ax1.scatter(radix_man_stats['bpm'], radix_hcv_stats['bpm'], label='$R. balthica$')
ax1.scatter(ciona_man_stats['bpm'], ciona_hcv_stats['bpm'], label='$C. intestinalis$')
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
print(m_bpm)

r, p = corr_stats(*pearsonr(m_bpm, h_bpm), len(m_bpm))
min, max = lims
ax1.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=8)
ax1.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=8)
ax1.set_xlabel('Manual heart rate (bpm)', fontsize=8)
ax1.set_ylabel('HeartCV heart rate (bpm)', fontsize=8)

ax2.text(-0.1, 1.1, "B", transform=ax2.transAxes, size=14, weight='bold')
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
ax2.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=8)
ax2.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=8)
ax2.set_xlabel('Manual minimum IBI (seconds)', fontsize=8)
ax2.set_ylabel('HeartCV minimum IBI (seconds)', fontsize=8)

ax3.text(-0.1, 1.1, "C", transform=ax3.transAxes, size=14, weight='bold')
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
ax3.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=8)
ax3.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=8)
ax3.set_xlabel('Manual mean IBI (seconds)', fontsize=8)
ax3.set_ylabel('HeartCV mean IBI (seconds)', fontsize=8)

ax4.text(-0.1, 1.1, "D", transform=ax4.transAxes, size=14, weight='bold')
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
ax4.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=8)
ax4.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=8)
ax4.set_xlabel('Manual maximum IBI (seconds)', fontsize=8)
ax4.set_ylabel('HeartCV maximum IBI (seconds)', fontsize=8)

ax5.text(-0.1, 1.1, "E", transform=ax5.transAxes, size=14, weight='bold')
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
ax5.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=8)
ax5.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=8)
ax5.set_xlabel('Manual σ IBI (seconds)', fontsize=8)
ax5.set_ylabel('HeartCV σ IBI (seconds)', fontsize=8)

ax6.text(-0.1, 1.1, "F", transform=ax6.transAxes, size=14, weight='bold')
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
ax6.text(max-(0.3*max), max-(max-(0.2*max)), f'{r}', size=8)
ax6.text(max-(0.3*max), max-(max-(0.15*max)), f'{p}', size=8)
ax6.set_xlabel('Manual RMSSD (seconds)', fontsize=8)
ax6.set_ylabel('HeartCV RMSSD (seconds)', fontsize=8)

plt.show()
# plt.savefig(f'{out_dir}/{result}.png')
# plt.clf()