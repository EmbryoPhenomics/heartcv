import heartcv as hcv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from heartcv.core.segmentation import _minmax_scale as scale


def gen_signal(file):
    hcv.show_progress(True)

    video = hcv.Video(file)
    frame = video.read(0)

    frames = video.read(low_memory=False)

    loc = hcv.Location(hcv.binary_thresh, hcv.largest)
    contour, _ = hcv.location_gui(video, loc)

    mask = hcv.contour_mask(frame, contour)
    mask = hcv.shrink(np.ones_like(frame), by=50)
    diff_img = hcv.sum_abs_diff(frames, mask)

    bbox, _ = hcv.activity_gui(frame, diff_img)
    frames.subset(*bbox)

    diff_frames = hcv.abs_diffs(frames)
    diff_vals = scale([np.sum(diff) for diff in diff_frames])

    video.close()

    return diff_vals


def parse(
    man, auto, up_to=100, every=2, left=0, inv_auto=False, rolling=False, win_size=3
):
    man = pd.read_csv(man)[:up_to]
    auto = pd.read_csv(auto)[:up_to]

    man_areas = man["area"][:up_to]
    auto_areas = auto["area"][:up_to]

    man_areas = [n for i, n in enumerate(man_areas) if i % every == left]
    auto_areas = [n for i, n in enumerate(auto_areas) if i % every == left]

    if inv_auto:
        auto_areas = max(auto_areas) - np.asarray(auto_areas)

    if rolling:
        auto_df = pd.DataFrame(data=dict(x=auto_areas))
        auto_areas = auto_df["x"].rolling(win_size).mean()

    man_areas, auto_areas = map(scale, (man_areas, auto_areas))

    return (man_areas.tolist(), auto_areas.tolist())


def sv(dia, sys):
    dia, sys = map(np.asarray, (dia, sys))
    sv = dia - sys
    return sv


def stats(sv):
    return [np.min(sv), np.max(sv), np.mean(sv), np.median(sv), np.std(sv)]


def scatter(x, y, xlabel, ylabel, path):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.show()


def append_with(to, with_):
    new = []
    for lt, lw in zip(to, with_):
        if not isinstance(lw, list):
            lw = [lw]
        new.append(lt + lw)
    return new


# Paleomon

source_dir = "/home/z/Documents/heartcv_testdata/mean_px/paleomon/raw/"

m_min_sv = []
m_max_sv = []
m_mean_sv = []
m_med_sv = []
m_std_sv = []
m_sv = []

all_m = [m_min_sv, m_max_sv, m_mean_sv, m_med_sv, m_std_sv]

a_min_sv = []
a_max_sv = []
a_mean_sv = []
a_med_sv = []
a_std_sv = []
a_sv = []

all_a = [a_min_sv, a_max_sv, a_mean_sv, a_med_sv, a_std_sv]

man, auto = parse(
    "./data/paleomon/sv_man_15_15ppt_young_1(1).csv",
    "./data/paleomon/sv_auto_15_15ppt_young_1.csv",
    100,
    1,
    0,
    False,
)
t, d, s = hcv.find_events(man, prominence=0.3)

auto = gen_signal(source_dir + "15_15ppt_young_1.avi")[:100]
a_events = hcv.find_events(auto, prominence=0.2)
sv_f = [n for i, n in enumerate(a_events[1][0]) if i % 2 == 0]

plt.plot(man)
plt.plot(*d, "x")
plt.plot(*s, "x")
plt.plot(auto)
plt.plot(sv_f, auto[sv_f], "x")
plt.show()

man_sv = d[1] - s[1]
auto_sv = auto[sv_f]

all_a = append_with(all_a, stats(auto_sv))
all_m = append_with(all_m, stats(man_sv))
m_sv = m_sv + man_sv.tolist()
a_sv = a_sv + auto_sv.tolist()

man, auto = parse(
    "./data/paleomon/sv_man_15_15ppt_young_C12_1.csv",
    "./data/paleomon/sv_auto_15_15ppt_young_C12_1.csv",
    100,
    2,
    0,
    True,
)
t, d, s = hcv.find_events(man, prominence=0.2)

auto = gen_signal(source_dir + "15_15ppt_young_C12_1.avi")[:100]
a_events = hcv.find_events(auto, prominence=0.2)
sv_f = [n for i, n in enumerate(a_events[1][0]) if i % 2 == 0]

plt.plot(man)
plt.plot(*d, "x")
plt.plot(*s, "x")
plt.plot(auto)
plt.plot(sv_f, auto[sv_f], "x")
plt.show()

man_sv = d[1] - s[1]
auto_sv = auto[sv_f]

all_a = append_with(all_a, stats(auto_sv))
all_m = append_with(all_m, stats(man_sv))
m_sv = m_sv + man_sv.tolist()
a_sv = a_sv + auto_sv.tolist()

man, auto = parse(
    "./data/paleomon/sv_manz_15_15ppt_young_A1_37.csv",
    "./data/paleomon/sv_auto_15_15ppt_young_A1_37.csv",
    53,
    1,
    0,
    True,
)
t, d, s = hcv.find_events(man, prominence=0.3)

auto = gen_signal(source_dir + "15_15ppt_young_A1_37.avi")[:53]
a_events = hcv.find_events(auto, prominence=0.2)
sv_f = [n for i, n in enumerate(a_events[1][0]) if i % 2 == 0]

plt.plot(man)
plt.plot(*d, "x")
plt.plot(*s, "x")
plt.plot(auto)
plt.plot(sv_f, auto[sv_f], "x")
plt.show()

man_sv = d[1] - s[1][1:]
auto_sv = auto[sv_f][1:]

all_a = append_with(all_a, stats(auto_sv))
all_m = append_with(all_m, stats(man_sv))
m_sv = m_sv + man_sv.tolist()
a_sv = a_sv + auto_sv.tolist()

man, auto = parse(
    "./data/paleomon/sv_man_15_15ppt_young_A4_60.csv",
    "./data/paleomon/sv_auto_15_15ppt_young_A4_60.csv",
    100,
    2,
    0,
    False,
)
t, d, s = hcv.find_events(man, prominence=0.3)

auto = gen_signal(source_dir + "15_15ppt_young_A4_60.avi")[:100]
a_events = hcv.find_events(auto, prominence=0.2)
sv_f = [n for i, n in enumerate(a_events[1][0]) if i % 2 == 0]

plt.plot(man)
plt.plot(*d, "x")
plt.plot(*s, "x")
plt.plot(auto)
plt.plot(sv_f, auto[sv_f], "x")
plt.show()

man_sv = d[1] - s[1][1:]
auto_sv = auto[sv_f][1:]

all_a = append_with(all_a, stats(auto_sv))
all_m = append_with(all_m, stats(man_sv))
m_sv = m_sv + man_sv.tolist()
a_sv = a_sv + auto_sv.tolist()

data = pd.read_csv("./data/paleomon/sve_man_15_25ppt_young_1.csv")
data = data.loc[data["EndSystoleFrame"] <= 100]
d = data["EndDiastoleArea"]
s = data["EndSystoleArea"]

plt.plot(data["EndDiastoleArea"], "x")
plt.plot(data["EndSystoleArea"], "x")
plt.show()

auto = gen_signal(
    "/home/z/Documents/heartcv_testdata/heartcv_testdata_1/15_25ppt_young_1.avi"
)[:100]
a_events = hcv.find_events(auto, prominence=0.2)
sv_f = [n for i, n in enumerate(a_events[1][0]) if i % 2 == 0]

plt.plot(auto)
plt.plot(sv_f, auto[sv_f], "x")
plt.show()

man_sv = scale(d - s)
auto_sv = auto[sv_f]

all_a = append_with(all_a, stats(auto_sv))
all_m = append_with(all_m, stats(man_sv))
m_sv = m_sv + man_sv.tolist()
a_sv = a_sv + auto_sv.tolist()

data = pd.read_csv("./data/paleomon/sve_man_15_35ppt_old_1.csv")
data = data.loc[data["EndSystoleFrame"] <= 100]
d = data["EndDiastoleArea"]
s = data["EndSystoleArea"]

plt.plot(data["EndDiastoleArea"], "x")
plt.plot(data["EndSystoleArea"], "x")
plt.show()

auto = gen_signal(
    "/home/z/Documents/heartcv_testdata/heartcv_testdata_1/15_35ppt_old_1.avi"
)[:100]
a_events = hcv.find_events(auto, prominence=0.01)
sv_f = [n for i, n in enumerate(a_events[1][0][1:]) if i % 2 == 0]

plt.plot(auto)
plt.plot(sv_f, auto[sv_f], "x")
plt.show()

man_sv = scale(d - s)
auto_sv = auto[sv_f]

all_a = append_with(all_a, stats(auto_sv))
all_m = append_with(all_m, stats(man_sv))
m_sv = m_sv + man_sv.tolist()
a_sv = a_sv + auto_sv.tolist()

data = pd.read_csv("./data/paleomon/sve_man_15_40ppt_medium_1.csv")
data = data.loc[data["EndSystoleFrame"] <= 100]
d = data["EndDiastoleArea"]
s = data["EndSystoleArea"]

plt.plot(data["EndDiastoleArea"], "x")
plt.plot(data["EndSystoleArea"], "x")
plt.show()

auto = gen_signal(
    "/home/z/Documents/heartcv_testdata/heartcv_testdata_1/15_40ppt_medium_1.avi"
)[:100]
a_events = hcv.find_events(auto, prominence=0.2)
sv_f = [n for i, n in enumerate(a_events[1][0]) if i % 2 == 0]

plt.plot(auto)
plt.plot(sv_f, auto[sv_f], "x")
plt.show()

man_sv = scale(d - s)
auto_sv = auto[sv_f]

all_a = append_with(all_a, stats(auto_sv))
all_m = append_with(all_m, stats(man_sv))
m_sv = m_sv + man_sv.tolist()
a_sv = a_sv + auto_sv.tolist()

m_min_sv, m_max_sv, m_mean_sv, m_med_sv, m_std_sv = map(scale, all_m)
a_min_sv, a_max_sv, a_mean_sv, a_med_sv, a_std_sv = all_a

# Stroke volume stats
fig, (
    (
        ax1,
        ax2,
        ax3,
    ),
    (ax4, ax5, ax6),
) = plt.subplots(2, 3)
ax1.scatter(m_min_sv, a_min_sv)
ax1.set_xlabel("Manual min stroke volume")
ax1.set_ylabel("HeartCV min stroke volume")

ax2.scatter(m_max_sv, a_max_sv)
ax2.set_xlabel("Manual max stroke volume")
ax2.set_ylabel("HeartCV max stroke volume")

ax3.scatter(m_mean_sv, a_mean_sv)
ax3.set_xlabel("Manual mean stroke volume")
ax3.set_ylabel("HeartCV mean stroke volume")

ax4.scatter(m_med_sv, a_med_sv)
ax4.set_xlabel("Manual med stroke volume")
ax4.set_ylabel("HeartCV med stroke volume")

ax5.scatter(m_std_sv, a_std_sv)
ax5.set_xlabel("Manual std stroke volume")
ax5.set_ylabel("HeartCV std stroke volume")

plt.show()

print(m_sv)

plt.scatter(scale(m_sv), a_sv)
plt.show()
