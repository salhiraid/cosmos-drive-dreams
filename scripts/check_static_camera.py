"""
Check whether generated videos keep the camera static.

Tracks corner features through each video and measures how far the background drifts from the first frame.
With a static camera, most tracked points (road, markings, poles, signs, buildings, sky) stay put, so the median
displacement stays near zero even when vehicles move through the frame. If the model moves the camera instead,
nearly every point shifts and the median drift grows.

Example:
    python scripts/check_static_camera.py -i outputs/parked_videos
    python scripts/check_static_camera.py -i outputs/parked_videos --threshold 8 --move_to outputs/parked_videos_moving
"""
import csv
import shutil
from pathlib import Path

import click
import cv2
import numpy as np

WORK_WIDTH = 640  # analysis resolution; results are reported in pixels of the original video width


def background_drift(video_path, max_corners=400, reseed_every=30):
    """
    Returns (max_drift, final_drift, max_step) in original-resolution pixels:
    max / final median displacement of tracked points relative to the first frame, and the largest median
    frame-to-frame motion (camera shake).
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {video_path}")
    scale = WORK_WIDTH / frame.shape[1]
    to_gray = lambda f: cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                                     cv2.COLOR_BGR2GRAY)

    prev = to_gray(frame)
    points = cv2.goodFeaturesToTrack(prev, max_corners, qualityLevel=0.01, minDistance=7)
    if points is None:
        return 0.0, 0.0, 0.0
    origin = points.copy()      # where each tracked point started (in first-frame coordinates)
    offset = np.zeros((len(points), 1, 2), np.float32)  # accumulated drift at reseed time
    drifts, steps, frame_idx = [0.0], [0.0], 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = to_gray(frame)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, points, None, winSize=(21, 21), maxLevel=3)
        good = status.ravel() == 1
        if good.sum() < 10:
            break
        step = np.linalg.norm((new_points - points)[good].reshape(-1, 2), axis=1)
        steps.append(float(np.median(step)))
        points, origin, offset = new_points[good], origin[good], offset[good]
        drifts.append(float(np.median(np.linalg.norm((points - origin + offset).reshape(-1, 2), axis=1))))

        # re-detect features now and then so the analysis does not run out of points; carry the current
        # median drift over so the measurement stays relative to the first frame
        if frame_idx % reseed_every == 0:
            current_drift = np.median((points - origin + offset).reshape(-1, 2), axis=0)
            fresh = cv2.goodFeaturesToTrack(gray, max_corners, qualityLevel=0.01, minDistance=7)
            if fresh is not None:
                points, origin = fresh, fresh.copy()
                offset = np.tile(current_drift, (len(fresh), 1)).reshape(-1, 1, 2).astype(np.float32)
        prev = gray

    cap.release()
    return max(drifts) / scale, drifts[-1] / scale, max(steps) / scale


@click.command()
@click.option("--input", "-i", "inputs", multiple=True, required=True, help="video file(s) or folder(s) of .mp4 files")
@click.option("--threshold", type=float, default=8.0,
              help="max background drift in pixels (original resolution) for a video to count as static")
@click.option("--csv_path", type=str, default=None, help="where to write the per-video report (default: next to the input)")
@click.option("--move_to", type=str, default=None, help="move videos whose camera moved into this folder")
def main(inputs, threshold, csv_path, move_to):
    videos = []
    for item in inputs:
        p = Path(item)
        videos += sorted(p.glob("*.mp4")) if p.is_dir() else [p]
    if not videos:
        raise click.UsageError("no .mp4 videos found")

    rows = []
    for video in videos:
        max_drift, final_drift, max_step = background_drift(video)
        static = max_drift <= threshold
        rows.append({"video": str(video), "static": static, "max_drift_px": round(max_drift, 2),
                     "final_drift_px": round(final_drift, 2), "max_step_px": round(max_step, 2)})
        print(f"{'STATIC' if static else 'MOVING'}  drift max {max_drift:7.2f}px  final {final_drift:7.2f}px  "
              f"max step {max_step:5.2f}px  {video.name}")
        if move_to and not static:
            Path(move_to).mkdir(parents=True, exist_ok=True)
            for f in video.parent.glob(video.stem + ".*"):  # the video and its prompt .txt
                shutil.move(str(f), str(Path(move_to) / f.name))

    csv_path = csv_path or str((Path(inputs[0]) if Path(inputs[0]).is_dir() else Path(inputs[0]).parent) /
                               "static_camera_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n_static = sum(r["static"] for r in rows)
    print(f"\n{n_static}/{len(rows)} videos have a static camera (threshold {threshold}px). Report: {csv_path}")


if __name__ == "__main__":
    main()
