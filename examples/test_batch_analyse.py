import importlib.util
from pathlib import Path
from types import SimpleNamespace
import csv

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('batch', Path(__file__).with_name('batch_analyse.py'))
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_ranges():
    assert batch.ranges(1250, 600) == [(0, 600), (600, 1200), (1200, 1250)]
    assert batch.ranges(1200, 600) == [(0, 600), (600, 1200)]
    assert batch.ranges(30, None) == [(0, 30)]


def test_file_list(tmp_path):
    listing = tmp_path / 'list.txt'
    listing.write_text('# comment\n"one video.avi"\n\none video.avi\nsecond.avi\n')
    assert batch.video_list(listing) == [tmp_path / 'one video.avi', tmp_path / 'second.avi']


def test_atomic_failure_keeps_previous(tmp_path):
    path = tmp_path / 'results.csv'
    batch.save_rows(path, [])
    before = path.read_bytes()
    def fail(stream):
        stream.write('partial')
        raise RuntimeError('simulated interruption')
    try:
        batch.atomic_write(path, fail)
    except RuntimeError:
        pass
    assert path.read_bytes() == before


def test_pipeline_resume_and_exports(tmp_path):
    video = tmp_path / 'test.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 32))
    assert writer.isOpened()
    for i in range(185):
        brightness = int(120 + 80 * np.sin(2 * np.pi * i / 15))
        writer.write(np.full((32, 32, 3), brightness, dtype=np.uint8))
    writer.release()
    args = SimpleNamespace(video=video, file_list=None, sequence_length=60, binsize=2, output=tmp_path/'results.csv')
    calls = []
    def selector(frames, fps, label, binsize):
        calls.append(label)
        action = ['accept', 'skip', 'stop'][len(calls)-1]
        return action, np.array([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.int32), 2.0
    batch.run(args, selector)
    def rows():
        with args.output.open() as stream:
            return list(csv.DictReader(stream))
    result = rows()
    assert [r['status'] for r in result] == ['ok', 'skipped', 'interrupted']
    assert float(result[0]['bpm']) > 0
    with open(result[0]['mpx_csv']) as stream:
        signal = list(csv.DictReader(stream))
    assert len(signal) == 60
    assert sum(int(r['is_beat']) for r in signal) == int(result[0]['beat_count'])
    assert signal[-1]['frame_index'] == '59'
    assert Path(result[0]['beat_plot']).stat().st_size > 1000
    resumed = []
    def skip(frames, fps, label, binsize):
        resumed.append(label)
        return 'skip', None, None
    batch.run(args, skip)
    assert len(resumed) == 2
    assert [r['status'] for r in rows()] == ['ok', 'skipped', 'skipped', 'skipped']
    assert rows()[-1]['frame_stop_exclusive'] == '185'
    batch.run(args, lambda *args: (_ for _ in ()).throw(AssertionError('must not rerun')))
    assert len(rows()) == 4


def test_missing_video_records_error(tmp_path):
    args = SimpleNamespace(video=tmp_path/'missing.avi', file_list=None, sequence_length=60, binsize=2, output=tmp_path/'results.csv')
    batch.run(args)
    with args.output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['status'] == 'error'
    assert list(rows[0])[0] == 'video_path'



def test_output_directory_and_resume(tmp_path):
    args = SimpleNamespace(video=tmp_path/'missing.avi', file_list=None,
                           sequence_length=600, binsize=2, output=tmp_path)
    batch.run(args)
    output = tmp_path / 'batch_results.csv'
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]['status'] == 'error'
    assert (tmp_path / 'mpx').is_dir()
    batch.run(args)
    with output.open() as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_start_frame_ranges():
    assert batch.ranges(1250, 600, 100) == [(100, 700), (700, 1250)]
    assert batch.ranges(1250, None, 100) == [(100, 1250)]
    assert batch.ranges(1250, 600, 1250) == []


def test_start_frame_pipeline(tmp_path):
    video = tmp_path / 'offset.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 32))
    assert writer.isOpened()
    for i in range(40):
        writer.write(np.full((32, 32, 3), i * 5, dtype=np.uint8))
    writer.release()
    args = SimpleNamespace(video=video, file_list=None, sequence_length=15,
                           start_frame=10, binsize=2, output=tmp_path/'results.csv')
    seen = []
    def select(frames, fps, label, binsize):
        seen.append((len(frames), float(frames[0].mean())))
        return 'accept', np.array([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.int32), 1.0
    batch.run(args, select)
    with args.output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(r['frame_start'], r['frame_stop_exclusive']) for r in rows] == [('10', '25'), ('25', '40')]
    assert seen == [(15, 50.0), (15, 125.0)]
    with open(rows[0]['mpx_csv']) as stream:
        signal = list(csv.DictReader(stream))
    assert signal[0]['frame_index'] == '10'
    assert float(signal[0]['video_time_seconds']) == 10 / 30
    args.start_frame = 40
    batch.run(args, select)
    with args.output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]['status'] == 'error'
    assert 'outside video' in rows[-1]['message']


def test_frequency_slider_computes_and_renders_without_callback(monkeypatch):
    positions = {'Frames': 0, 'Frequency index': 0}
    displays, calls = [], []
    labels = []
    original_put_text = cv2.putText
    def record_text(image, text, *args, **kwargs):
        if image.shape[0] == 55:
            labels.append(text)
        return original_put_text(image, text, *args, **kwargs)
    monkeypatch.setattr(cv2, 'putText', record_text)
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', lambda *a: None)
    monkeypatch.setattr(cv2, 'createTrackbar', lambda *a: None)
    monkeypatch.setattr(cv2, 'getTrackbarPos', lambda name, title: positions[name])
    monkeypatch.setattr(cv2, 'imshow', lambda title, image: displays.append(image.copy()))
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *a: 1)
    count = [0]
    def key(*args):
        count[0] += 1
        if count[0] == 1:
            positions['Frequency index'] = 1
            return -1
        import time
        time.sleep(.005)
        assert count[0] < 400, 'GUI did not render background result'
        if not any(np.any(image[95:-55, 300:]) for image in displays):
            return -1
        if positions['Frequency index'] == 1:
            positions['Frequency index'] = 2
            return -1
        return 13
    monkeypatch.setattr(cv2, 'waitKey', key)
    def epts(*args, **kwargs):
        calls.append(True)
        freq = np.broadcast_to((np.arange(6) / 4)[:, None, None], (6, 16, 16)).copy()
        power = np.zeros((6, 16, 16))
        power[1, 2:6, 2:6] = 1
        power[2, 8:12, 8:12] = 1
        return freq, power
    monkeypatch.setattr(batch.hcv, 'epts', epts)
    action, roi, frequency = batch.select_region(np.zeros((10, 32, 32), dtype=np.uint8), 30, 'test', 2)
    assert action == 'accept' and frequency == 0.5 and roi is not None
    assert set(labels) == {'Hz', '0', '1'}
    assert len(calls) == 1
    maps = [image[95:-55, 300:] for image in displays if np.any(image[95:-55, 300:])]
    assert len(maps) >= 2
    assert not np.array_equal(maps[-2], maps[-1])


def test_vuba_skip_button(monkeypatch):
    callback = {}
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', lambda title, cb: callback.update(mouse=cb))
    monkeypatch.setattr(cv2, 'createTrackbar', lambda *a: None)
    monkeypatch.setattr(cv2, 'getTrackbarPos', lambda *a: 0)
    monkeypatch.setattr(cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *a: 1)
    monkeypatch.setattr(batch.hcv, 'epts', lambda *a, **kw: (np.zeros((6, 2, 2)), np.zeros((6, 2, 2))))
    def click(*args):
        callback['mouse'](cv2.EVENT_LBUTTONDOWN, 300, 20, 0, None)
        return -1
    monkeypatch.setattr(cv2, 'waitKey', click)
    assert batch.select_region(np.zeros((10, 32, 32), dtype=np.uint8), 30, 'test', 16) == ('skip', None, None)


def test_fps_override_outputs_and_resume(tmp_path):
    video = tmp_path / 'fps.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 32))
    assert writer.isOpened()
    for i in range(60):
        writer.write(np.full((32, 32, 3), int(120 + 80 * np.sin(2 * np.pi * i / 15)), dtype=np.uint8))
    writer.release()
    args = SimpleNamespace(video=video, file_list=None, sequence_length=60,
                           fps=15.0, binsize=2, output=tmp_path/'results.csv')
    seen = []
    def select(frames, fps, label, binsize):
        seen.append(fps)
        return 'accept', np.array([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.int32), 1.0
    batch.run(args, select)
    batch.run(args, select)
    assert seen == [15.0]
    args.fps = 30.0
    batch.run(args, select)
    assert seen == [15.0, 30.0]
    with args.output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(r['stop_seconds']) for r in rows] == [4.0, 2.0]
    assert float(rows[1]['bpm']) == 2 * float(rows[0]['bpm'])
    assert rows[0]['item_id'] != rows[1]['item_id']
    with open(rows[0]['mpx_csv']) as stream:
        signal = list(csv.DictReader(stream))
    assert float(signal[15]['video_time_seconds']) == 1.0


def test_invalid_fps(tmp_path):
    import pytest
    for fps in (0, -1, float('nan'), float('inf')):
        args = SimpleNamespace(video=tmp_path/'unused.avi', file_list=None,
                               fps=fps, output=tmp_path/'results.csv')
        with pytest.raises(ValueError, match='FPS'):
            batch.run(args)
