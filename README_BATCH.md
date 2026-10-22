# Batch and segmented HeartCV analysis

Activate the environment and run from the repository root:

```bash
conda activate heartcv
cd /Users/otills/Github/heartcv
```

## List of videos

Create a UTF-8 text file with one video path per line. Paths containing spaces are supported, with or without surrounding quotes. Blank lines and lines starting with `#` are ignored. Relative paths are resolved relative to the list file. Duplicate paths are analysed once.

```text
/path/to/video_1.avi
/path/to/video 2.mp4
```

```bash
python examples/batch_analyse.py --file-list videos.txt --output results/batch.csv
```

## Long video, in frames

```bash
python examples/batch_analyse.py --video /path/to/long_video.avi --sequence-length 600 --output results/segments.csv
```

`--sequence-length` is a **positive integer number of frames**, not seconds. Segments do not overlap: `[0, 600)`, `[600, 1200)`, and so on. Frame indices start at zero; the stop index is exclusive. The final shorter segment is included. Segments shorter than four frames are recorded as `too_short`. You can also combine `--file-list` with `--sequence-length` to segment every listed video. Without a sequence length each video is one item. Only the current segment is loaded into memory; choose manageable segments for long recordings (HeartCV's peak detection also uses substantial memory).

Use `--start-frame 1200` to begin at frame 1200 (zero-based). With `--sequence-length 600`, the first segment is `[1200, 1800)`. Without a sequence length, all remaining frames are analysed as one item. This applies to every video in a file list. Output indices and times remain relative to the original video. A start frame at or beyond the end is recorded as an error and processing continues to the next video.

```bash
python examples/batch_analyse.py --video /path/to/long_video.avi --start-frame 1200 --sequence-length 600 --output results/segments.csv
```

Use `--fps 30` (or a fractional value such as `--fps 29.97`) to override the source frame rate. The override applies to every video in a file list and is used for EPT frequencies, cardiac statistics, CSV timestamps and MPX plots. It does not resample frames or change segment boundaries. Without it, the source frame rate is used. Changing the override creates new analysis identifiers, so results calculated at another FPS are not silently reused.

## Manual selection with Vuba

The script reads each segment using `vuba.Video.read`, computes EPTs before opening the window, and uses Vuba's `FramesGUI` for the frame/frequency sliders. The default `--binsize` is now **16**, matching the multiple-timepoint example; specify `--binsize 2` if you need the previous finer spatial map. Changing binsize causes saved segments to be treated as new analyses.

Adjust the Frequency index slider to locate the heart. A ruler below the images labels whole Hz values (0, 1, 2, …), with a green marker at the selected frequency. Intermediate frequency bins remain selectable without ruler labels; the precise selected frequency appears above the images. Click **Accept** or press Enter to analyse, save and advance. Click **Skip** (S) if the heart is absent or invisible. Click **Stop** (Q/Escape), or close the window, to save and exit. EPT computation happens before the window appears; Ctrl+C can interrupt it. Frequency changes reuse the computed spectra. No C key is needed.

Signal processing remains unchanged: no inversion or interpolation is applied. Skipped items have no MPX or beat plot because no region was accepted. The selected region and detected beats should be checked by the user.

## Saved results and resume

`--output` accepts a CSV filename or an existing folder. For example, `--output ~/Desktop/` saves `~/Desktop/batch_results.csv` and puts signal files in `~/Desktop/mpx/`. To use a new folder, specify the full CSV path; its parent folders will be created.

The results CSV's first column is the full absolute video path including its filename. Separate filename, frame range, time range, frame rate, selected frequency, region coordinates, beat count, status, statistics and output paths are included.

The CSV is atomically replaced after **each** completed, skipped, failed or interrupted item. If the application crashes, all previously saved items remain; the active item may need to be repeated. Run the same command with the same output path to resume. Completed, skipped and error items are not repeated; interrupted items are. To intentionally reanalyse everything, choose a new output CSV. To retry selected errors/skips, remove their rows from the CSV while the script is stopped. Do not run two instances against the same output CSV simultaneously.

Source path, source file size/modification time, frame range and binsize form the item identifier, so changed source files or segmentation settings are not mistaken for completed items. Keep the original source files unchanged for straightforward resume.

Alongside the results CSV, an `mpx/` folder contains a CSV and PNG for each measured item. Names include the video stem, frame range and a unique identifier. Each MPX row contains the original source path, segment bounds, absolute frame index, segment/video times, MPX value, beat flag and interval since the previous beat. The plot marks the exact detected peaks used for the statistics, and its x-axis is time in the source video.

MPX follows the original example's convention: the full-frame mean after masking outside the selected rectangle to zero. The signal is not interpolated, and statistics use the **actual frame rate** and sample count (no factor-of-three adjustment). Fewer than two detected beats produces `insufficient_beats` with empty statistics, rather than a zero heart rate. Beat-to-beat intervals are in seconds. Frame-based times assume constant frame rate; variable-frame-rate recordings should first be converted to constant frame rate.
