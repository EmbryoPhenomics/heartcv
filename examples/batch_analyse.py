#!/usr/bin/env python3
"""Resumable, manually supervised HeartCV batch analysis. See README_BATCH.md."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import cv2
import vuba
import numpy as np
import heartcv as hcv
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

STATS = ['bpm', 'min_b2b', 'mean_b2b', 'median_b2b', 'max_b2b', 'sd_b2b', 'range_b2b', 'rmssd']
FIELDS = ['video_path', 'filename', 'frame_start', 'frame_stop_exclusive', 'start_seconds',
          'stop_seconds', 'fps', 'status', 'message', 'frequency_hz', 'beat_count',
          'roi_json', 'mpx_csv', 'beat_plot', 'item_id'] + STATS


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_rows(path, rows):
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(path, write)


def video_list(path):
    path = Path(path).expanduser().resolve()
    result = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
            line = line[1:-1]
        video = Path(line).expanduser()
        result.append((path.parent / video).resolve())
    return list(dict.fromkeys(result))


def ranges(total, frames_per_segment, start_frame=0):
    size = total if frames_per_segment is None else frames_per_segment
    return [(start, min(start + size, total)) for start in range(start_frame, total, size)]


def identity(path, start, stop, binsize, fps_override=None):
    stat = path.stat() if path.exists() else None
    value = [str(path), stat.st_size if stat else None, stat.st_mtime_ns if stat else None,
             start, stop, binsize]
    if fps_override is not None:
        value.append(float(fps_override))
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:20]


def select_region(frames, fps, label, binsize):
    """Vuba frame/frequency sliders with explicit accept, skip and stop controls."""
    epts = hcv.epts(frames, fs=fps, binsize=binsize)
    title = 'HeartCV batch - Vuba frequency selection'
    state = {'action': None, 'roi': None, 'frequency': None}
    height, width = frames.shape[1:]
    display_width = max(300, min(width, 550))
    display_height = max(150, round(height * display_width / width))
    canvas_width = display_width * 2
    gui = vuba.FramesGUI(frames, title=title)
    gui.frame = frames[0]

    @gui.method
    def locate(gui):
        fi = gui['freq_ind'] if 'freq_ind' in gui.trackbars else 0
        state['frequency'] = float(epts[0][fi, 0, 0])
        frame = cv2.cvtColor(gui.frame, cv2.COLOR_GRAY2BGR)
        power = epts[1][fi]
        heatmap = np.zeros((height, width), dtype=np.uint8)
        state['roi'] = None
        if np.isfinite(power).all() and np.ptp(power) > 0:
            scaled = ((power - power.min()) / np.ptp(power) * 255).astype(np.uint8)
            heatmap = cv2.resize(scaled, (width, height))
            contour = hcv.detect_largest(heatmap)
            state['roi'] = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
            cv2.drawContours(frame, [state['roi']], -1, (0, 255, 0), 1)
        body = np.hstack([cv2.resize(frame, (display_width, display_height)),
                          cv2.cvtColor(cv2.resize(heatmap, (display_width, display_height)), cv2.COLOR_GRAY2BGR)])
        bar = np.zeros((95, canvas_width, 3), dtype=np.uint8)
        for i, text in enumerate(('Accept [Enter]', 'Skip [S]', 'Stop [Q]')):
            x0, x1 = i * canvas_width // 3, (i + 1) * canvas_width // 3
            cv2.rectangle(bar, (x0 + 2, 2), (x1 - 2, 45), (65, 65, 65), -1)
            cv2.putText(bar, text, (x0 + 8, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(bar, 'Frequency: %.3f Hz' % state['frequency'], (8, 66), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(bar, label[-95:], (8, 86), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        # Label whole Hz positions without rounding or dropping selectable FFT bins.
        frequencies = epts[0][:, 0, 0]
        low, high = float(frequencies[0]), float(frequencies[-1])
        ruler = np.zeros((55, canvas_width, 3), dtype=np.uint8)
        left, right = 24, canvas_width - 24
        cv2.putText(ruler, 'Hz', (2, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        if high > low:
            def position(hz):
                return left + round((hz - low) / (high - low) * (right - left))
            cv2.line(ruler, (left, 20), (right, 20), (180, 180, 180), 1)
            for hz in range(int(np.ceil(low)), int(np.floor(high)) + 1):
                x = position(hz)
                text = str(hz)
                text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0]
                cv2.line(ruler, (x, 20), (x, 27), (255, 255, 255), 1)
                cv2.putText(ruler, text, (x - text_width // 2, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            selected = position(state['frequency'])
            cv2.line(ruler, (selected, 7), (selected, 19), (0, 255, 0), 2)
        return np.vstack([bar, body, ruler])

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= y < 48 and 0 <= x < canvas_width:
            state['action'] = ('accept', 'skip', 'stop')[min(2, x * 3 // canvas_width)]
    try:
        gui.trackbar('Frequency index', id='freq_ind', min=0, max=len(epts[0]) - 1)(None)
        cv2.setMouseCallback(title, mouse)
        # Vuba supplies the rendering callbacks; this loop keeps explicit button semantics.
        gui.read(gui, 0)
        last_positions = None
        while True:
            positions = (min(cv2.getTrackbarPos('Frames', title), len(frames) - 1),
                         cv2.getTrackbarPos('Frequency index', title))
            if positions != last_positions:
                gui['freq_ind'] = positions[1]
                gui.read(gui, positions[0])
                last_positions = positions
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return 'stop', None, None
            if key == ord('s'):
                state['action'] = 'skip'
            if key in (10, 13):
                state['action'] = 'accept'
            action, state['action'] = state['action'], None
            if action in ('skip', 'stop'):
                return action, None, None
            if action == 'accept' and state['roi'] is not None:
                return action, state['roi'], state['frequency']
    finally:
        cv2.destroyAllWindows()


def analyse(frames, fps, roi):
    # Match the existing example: rectangular mask, full-frame mean including masked zeros.
    signal = np.asarray([frame.mean() for frame in hcv.segment(frames, roi)], dtype=float)
    peaks = np.asarray(hcv.find_peaks(signal), dtype=int) if len(signal) >= 4 and np.ptp(signal) > 0 else np.array([], dtype=int)
    stats = {key: '' for key in STATS}
    if len(peaks) >= 2:
        stats.update({key: float(value[0]) for key, value in hcv.stats(peaks, len(signal), fps).items()})
    return signal, peaks, stats


def save_signal(folder, row, signal, peaks):
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(row['video_path']).stem)[:70]
    stem = '%s_f%s-%s_%s' % (slug, row['frame_start'], row['frame_stop_exclusive'], row['item_id'])
    csv_path, plot_path = folder / (stem + '.csv'), folder / (stem + '.png')
    peak_set = set(peaks.tolist())
    def write(stream):
        writer = csv.writer(stream)
        writer.writerow(['video_path', 'segment_start', 'segment_stop_exclusive', 'frame_index',
                         'segment_time_seconds', 'video_time_seconds', 'mpx', 'is_beat', 'interval_from_previous_beat_seconds'])
        previous = None
        for i, value in enumerate(signal):
            beat = i in peak_set
            interval = (i - previous) / row['fps'] if beat and previous is not None else ''
            writer.writerow([row['video_path'], row['frame_start'], row['frame_stop_exclusive'],
                             row['frame_start'] + i, i / row['fps'], (row['frame_start'] + i) / row['fps'],
                             value, int(beat), interval])
            if beat:
                previous = i
    atomic_write(csv_path, write)
    fig = Figure(figsize=(12, 4), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    time = (np.arange(len(signal)) + row['frame_start']) / row['fps']
    ax.plot(time, signal, color='black', linewidth=.8, label='MPX')
    ax.plot(time[peaks], signal[peaks], 'ro', markersize=4, label='Detected beats')
    ax.set(xlabel='Time in source video (seconds)', ylabel='Mean pixel value',
           title='%s | frames [%s, %s)' % (row['filename'], row['frame_start'], row['frame_stop_exclusive']))
    ax.legend()
    temp = plot_path.with_suffix('.tmp')
    fig.savefig(temp, format='png', dpi=150)
    os.replace(temp, plot_path)
    row.update(mpx_csv=str(csv_path), beat_plot=str(plot_path))


def run(args, selector=select_region):
    output = args.output.expanduser().resolve()
    if output.is_dir():
        output = output / 'batch_results.csv'
    if output.exists():
        with output.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError('Output CSV is not a compatible HeartCV batch results file. Choose a new output path.')
            rows = list(reader)
    else:
        rows = []
    done = {row['item_id'] for row in rows if row['status'] != 'interrupted'}
    videos = video_list(args.file_list) if args.file_list else [args.video.expanduser().resolve()]
    if not videos:
        raise ValueError('The file list contains no videos.')
    folder = output.parent / 'mpx'
    folder.mkdir(parents=True, exist_ok=True)
    fps_override = getattr(args, 'fps', None)
    if fps_override is not None and (not np.isfinite(fps_override) or fps_override <= 0):
        raise ValueError('FPS must be a positive finite number')
    start_frame = getattr(args, 'start_frame', 0)
    if start_frame < 0:
        raise ValueError('Start frame must be non-negative')
    for path in videos:
        capture = vuba.Video(str(path))
        try:
            fps = fps_override if fps_override is not None else capture.video.get(cv2.CAP_PROP_FPS)
            total = len(capture) if capture.video.isOpened() else 0
            valid = capture.video.isOpened() and np.isfinite(fps) and fps > 0 and total > 0
            jobs = ranges(total, args.sequence_length, start_frame) if valid else []
            if not jobs:
                jobs = [(start_frame, start_frame)]
            for start, stop in jobs:
                item_id = identity(path, start, stop, args.binsize, fps_override)
                if item_id in done:
                    print('Already saved:', path, start, stop)
                    continue
                row = dict(video_path=str(path), filename=path.name, frame_start=start,
                           frame_stop_exclusive=stop, fps=fps if valid else '', item_id=item_id,
                           start_seconds=start / fps if valid else '', stop_seconds=stop / fps if valid else '')
                try:
                    if not valid:
                        raise ValueError('Cannot open video or read a valid frame count/frame rate')
                    if start >= total:
                        raise ValueError('Start frame %d is outside video with %d frames' % (start, total))
                    print('\nAnalysing %s frames [%d, %d)' % (path, start, stop), flush=True)
                    frames = np.asarray(list(capture.read(start=start, stop=stop, grayscale=True)))
                    if len(frames) != stop - start or frames.ndim != 3:
                        raise ValueError('Video decoding failed within the requested frame range')
                    if len(frames) < 4:
                        row.update(status='too_short', message='Fewer than four frames; no rate calculated')
                    else:
                        action, roi, frequency = selector(frames, fps, '%s [%d, %d)' % (path.name, start, stop), args.binsize)
                        if action == 'stop':
                            raise KeyboardInterrupt
                        if action == 'skip':
                            row.update(status='skipped', message='Heart absent or not visible; skipped by user')
                        else:
                            row.update(frequency_hz=frequency, roi_json=json.dumps(roi.tolist()))
                            signal, peaks, stats = analyse(frames, fps, roi)
                            row.update(stats)
                            row.update(beat_count=len(peaks), status='ok' if len(peaks) >= 2 else 'insufficient_beats', message='')
                            save_signal(folder, row, signal, peaks)
                except KeyboardInterrupt:
                    row.update(status='interrupted', message='Stopped; this item will be offered again on resume')
                    rows = [old for old in rows if old['item_id'] != item_id] + [row]
                    save_rows(output, rows)
                    print('Progress saved:', output)
                    return
                except Exception as error:
                    row.update(status='error', message='%s: %s' % (type(error).__name__, error))
                    print(row['message'])
                rows = [old for old in rows if old['item_id'] != item_id] + [row]
                save_rows(output, rows)
                done.add(item_id)
                print('Saved:', row['status'], output, flush=True)
        finally:
            capture.close()
    print('Finished:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--file-list', type=Path, help='Text file: one video path per line')
    source.add_argument('--video', type=Path, help='One video to analyse')
    parser.add_argument('--sequence-length', type=int, help='Frames per segment (default: entire video)')
    parser.add_argument('--output', type=Path, default=Path('batch_results.csv'),
                        help='CSV filename or existing folder (uses batch_results.csv inside it)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='First frame to analyse, zero-based (default: 0); applies to each video')
    parser.add_argument('--fps', type=float, help='Override source frame rate for all timing and calculations')
    parser.add_argument('--binsize', type=int, default=16, help='EPT spatial downsampling (default: 16)')
    args = parser.parse_args()
    if args.sequence_length is not None and args.sequence_length <= 0:
        parser.error('--sequence-length must be a positive integer number of frames')
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be a positive finite number')
    if args.start_frame < 0:
        parser.error('--start-frame must be a non-negative integer')
    if args.binsize < 1:
        parser.error('--binsize must be at least 1')
    run(args)


if __name__ == '__main__':
    main()
