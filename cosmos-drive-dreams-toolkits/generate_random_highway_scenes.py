"""
Automatically generate many randomized fixed-camera highway clips in RDS-HQ format.

Every clip draws its own road layout, traffic density, speeds, vehicle mix, vehicle size variation and
camera placement (overhead gantry or roadside pole, looking with or against traffic), so a batch covers
a wide variety of scenes. The parameters of each clip are saved to `scene_params/<clip_id>.json` so any
clip can be reproduced with create_fixed_camera_highway.py, and `clip_ids.json` lists the generated clips.

Example:
    python generate_random_highway_scenes.py -o highway_random -n 20
    python render_from_rds_hq.py -i highway_random -o highway_random_render -d highway_fixed -c pinhole \
        -cj highway_random/clip_ids.json --skip lidar --skip world_scenario
"""
import json
from pathlib import Path

import click
import numpy as np

from create_fixed_camera_highway import DEFAULT_MIX, VEHICLE_CLASSES, create_scene, parked_camera, parse_mix

# traffic regimes: bumper-to-bumper gap range (m) and lane speed range (m/s)
TRAFFIC_LEVELS = {
    "free_flow": {"gap": (30.0, 80.0), "speed": (27.0, 36.0)},
    "moderate":  {"gap": (15.0, 40.0), "speed": (22.0, 30.0)},
    "dense":     {"gap": (5.0, 18.0),  "speed": (13.0, 22.0)},
    "jam":       {"gap": (2.0, 6.0),   "speed": (2.0, 8.0)},
}
CAMERA_MOUNTS = ["gantry", "pole", "parked", "ego"]


def sample_range(rng, low_high, min_width=0.0):
    """Random sub-range [a, b] inside low_high, at least min_width wide."""
    low, high = low_high
    a = rng.uniform(low, high - min_width)
    b = rng.uniform(a + min_width, high)
    return float(a), float(b)


def sample_mix(rng, concentration):
    """Randomize the default vehicle mix with a Dirichlet draw (lower concentration = more extreme mixes)."""
    base = parse_mix(DEFAULT_MIX)
    weights = rng.dirichlet([concentration * base[name] for name in VEHICLE_CLASSES])
    return {name: round(float(w), 4) for name, w in zip(VEHICLE_CLASSES, weights)}


def sample_camera(rng, mount, num_lanes, lane_width, median_width, shoulder_width, min_speed, max_speed):
    half_road = median_width / 2 + num_lanes * lane_width
    if mount == "ego":
        # front camera of an ego car in a lane (mostly the middle one), driving with traffic or stopped
        ego_lane = num_lanes // 2 if rng.random() < 0.6 else int(rng.integers(0, num_lanes))
        ego_speed = 0.0 if rng.random() < 0.4 else float(rng.uniform(min_speed, max_speed))
        return {
            "ego_lane": ego_lane,
            "ego_speed": round(ego_speed, 2),
            "cam_x": 0.0,
            "cam_y": 0.0,
            "cam_height": round(float(rng.uniform(1.3, 2.2)), 2),
            "yaw": 0.0,
            "pitch": round(float(rng.uniform(0.0, 3.0)), 2),
            "hfov": round(float(rng.uniform(90.0, 120.0)), 2),
        }
    if mount == "parked":
        # front camera of a car standing on the outer shoulder, like an ego-vehicle view
        side = str(rng.choice(["right", "left"]))
        facing = str(rng.choice(["with_traffic", "against_traffic"]))
        cam_y, yaw = parked_camera(side, facing, num_lanes, lane_width, median_width, shoulder_width)
        return {
            "cam_x": 0.0,
            "cam_y": round(float(cam_y + rng.uniform(-0.4, 0.4)), 2),
            "cam_height": round(float(rng.uniform(1.3, 2.2)), 2),
            "yaw": round(float(yaw + rng.uniform(-4.0, 4.0)), 2),
            "pitch": round(float(rng.uniform(0.0, 4.0)), 2),
            "hfov": round(float(rng.uniform(90.0, 120.0)), 2),
        }
    # look along traffic (+x) or against it (-x)
    base_yaw = 0.0 if rng.random() < 0.7 else 180.0
    if mount == "gantry":
        # above the road, anywhere across the carriageways, looking along the highway
        cam_y = rng.uniform(-half_road, half_road)
        cam_height = rng.uniform(6.5, 12.0)
        yaw_offset = rng.uniform(-8.0, 8.0)
        pitch = rng.uniform(12.0, 28.0)
    else:
        # on either shoulder, turned towards the road
        side = rng.choice([-1.0, 1.0])
        cam_y = side * (half_road + rng.uniform(2.0, 15.0))
        cam_height = rng.uniform(6.0, 20.0)
        # a camera on the -y side turns towards +y: positive yaw when looking +x, negative when looking -x
        turn = rng.uniform(10.0, 45.0) * -side
        yaw_offset = turn if base_yaw == 0.0 else -turn
        pitch = rng.uniform(10.0, 30.0)
    return {
        "cam_x": 0.0,
        "cam_y": round(float(cam_y), 2),
        "cam_height": round(float(cam_height), 2),
        "yaw": round(float(base_yaw + yaw_offset), 2),
        "pitch": round(float(pitch), 2),
        "hfov": round(float(rng.uniform(50.0, 90.0)), 2),
    }


def sample_scene(rng, min_lanes, max_lanes, traffic_levels, camera_mounts, mix_concentration,
                 min_size_variation, max_size_variation, max_zigzag_ratio, max_lane_change_rate):
    num_lanes = int(rng.integers(min_lanes, max_lanes + 1))
    lane_width = float(rng.uniform(3.3, 3.8))
    median_width = float(rng.uniform(0.6, 5.0))
    level = str(rng.choice(traffic_levels))
    min_gap, max_gap = sample_range(rng, TRAFFIC_LEVELS[level]["gap"], min_width=2.0)
    min_speed, max_speed = sample_range(rng, TRAFFIC_LEVELS[level]["speed"], min_width=1.0)
    mount = str(rng.choice(camera_mounts))
    # a car parked on the shoulder needs a wide shoulder, otherwise a narrow paved edge
    shoulder_width = float(rng.uniform(3.0, 4.5)) if mount == "parked" else float(rng.uniform(0.5, 2.5))

    return {
        "traffic_level": level,
        "camera_mount": mount,
        "num_lanes": num_lanes,
        "lane_width": round(lane_width, 2),
        "median_width": round(median_width, 2),
        "shoulder_width": round(shoulder_width, 2),
        # static roadside landmarks help the video model keep the camera still
        "pole_spacing": round(float(rng.uniform(25.0, 60.0)), 1),
        "sign_spacing": round(float(rng.uniform(80.0, 250.0)), 1),
        "min_gap": round(min_gap, 2),
        "max_gap": round(max_gap, 2),
        "min_speed": round(min_speed, 2),
        "max_speed": round(max_speed, 2),
        "mix": sample_mix(rng, mix_concentration),
        "size_variation": round(float(rng.uniform(min_size_variation, max_size_variation)), 3),
        # lane changes: some clips without, others with zigzagging and / or normal lane changes
        "zigzag_ratio": round(float(rng.uniform(0.0, max_zigzag_ratio)) if rng.random() < 0.7 else 0.0, 3),
        "zigzag_period": round(float(rng.uniform(0.9, 2.0)), 2),
        "zigzag_near": int(rng.integers(0, 4)),
        "lane_change_rate": round(float(rng.uniform(0.0, max_lane_change_rate)) if rng.random() < 0.7 else 0.0, 2),
        **sample_camera(rng, mount, num_lanes, lane_width, median_width, shoulder_width, min_speed, max_speed),
        "seed": int(rng.integers(0, 2**31 - 1)),
    }


@click.command()
@click.option("--output_root", "-o", type=str, required=True, help="output folder in RDS-HQ format")
@click.option("--num_clips", "-n", type=int, default=10, help="number of random clips to generate")
@click.option("--prefix", type=str, default="highway_random", help="clip id prefix")
@click.option("--num_frames", type=int, default=121, help="frames per clip at 30 fps")
@click.option("--min_lanes", type=int, default=2, help="min lanes per direction")
@click.option("--max_lanes", type=int, default=5, help="max lanes per direction")
@click.option("--traffic", type=str, default=",".join(TRAFFIC_LEVELS),
              help=f"comma-separated traffic levels to draw from: {', '.join(TRAFFIC_LEVELS)}")
@click.option("--camera", type=str, default=",".join(CAMERA_MOUNTS),
              help=f"comma-separated camera mounts to draw from: {', '.join(CAMERA_MOUNTS)}")
@click.option("--mix_concentration", type=float, default=20.0,
              help="how close each clip's vehicle mix stays to the default (lower = more variety)")
@click.option("--min_size_variation", type=float, default=0.1, help="min per-clip size variation")
@click.option("--max_size_variation", type=float, default=0.2, help="max per-clip size variation (0.2 = +-20%)")
@click.option("--max_zigzag_ratio", type=float, default=0.2,
              help="max fraction of light vehicles that zigzag between lanes (0 = never)")
@click.option("--max_lane_change_rate", type=float, default=2.0,
              help="max normal lane changes per vehicle per minute (0 = never)")
@click.option("--seed", type=int, default=0, help="master seed; the same seed gives the same batch")
def main(output_root, num_clips, prefix, num_frames, min_lanes, max_lanes, traffic, camera, mix_concentration,
         min_size_variation, max_size_variation, max_zigzag_ratio, max_lane_change_rate, seed):
    traffic_levels = [t.strip() for t in traffic.split(",")]
    camera_mounts = [c.strip() for c in camera.split(",")]
    for name, valid in [(traffic_levels, TRAFFIC_LEVELS), (camera_mounts, CAMERA_MOUNTS)]:
        unknown = set(name) - set(valid)
        if unknown:
            raise click.BadParameter(f"unknown value(s) {sorted(unknown)}, choose from {list(valid)}")

    rng = np.random.default_rng(seed)
    output_root_p = Path(output_root)
    (output_root_p / "scene_params").mkdir(parents=True, exist_ok=True)

    clip_ids = []
    for i in range(num_clips):
        clip_id = f"{prefix}_{i:04d}"
        params = sample_scene(rng, min_lanes, max_lanes, traffic_levels, camera_mounts, mix_concentration,
                              min_size_variation, max_size_variation,
                              max_zigzag_ratio, max_lane_change_rate)
        with open(output_root_p / "scene_params" / f"{clip_id}.json", "w") as f:
            json.dump(params, f, indent=2)

        scene_args = {k: v for k, v in params.items() if k not in ("traffic_level", "camera_mount")}
        create_scene(output_root, clip_id, num_frames=num_frames, **scene_args)
        clip_ids.append(clip_id)
        print(f"[{i + 1}/{num_clips}] {clip_id}: {params['num_lanes'] * 2} lanes, {params['traffic_level']} traffic, "
              f"{params['camera_mount']} camera")

    with open(output_root_p / "clip_ids.json", "w") as f:
        json.dump(clip_ids, f, indent=2)


if __name__ == "__main__":
    main()
