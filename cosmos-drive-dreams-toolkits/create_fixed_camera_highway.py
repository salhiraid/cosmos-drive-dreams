"""
Create a synthetic highway clip seen from a fixed (static) roadside camera, in RDS-HQ format.

The camera never moves, like a traffic / radar-mounted camera on a pole or gantry. Vehicles drive
through the scene on a straight multi-lane highway. The output can be rendered with
`render_from_rds_hq.py -d highway_fixed -c pinhole` to get HD map + 3D bounding box condition videos.

Output layout (under OUTPUT_ROOT):
    pose/<clip_id>.tar               camera-to-world (OpenCV convention), identical for every frame
    pinhole_intrinsic/<clip_id>.tar  [fx, fy, cx, cy, w, h]
    all_object_info/<clip_id>.tar    per-frame 3D cuboids of all vehicles
    3d_lanelines/<clip_id>.tar       lane divider polylines
    3d_road_boundaries/<clip_id>.tar road edge polylines

World frame: x along the highway, y to the left, z up. The road surface is z = 0.
"""
import click
import numpy as np

from utils.wds_utils import write_to_tar

CAMERA_NAME = "roadside_cam"
FPS = 30

# length, width, height in meters
OBJECT_SIZES = {
    "Car": [4.6, 1.9, 1.6],
    "Truck": [16.0, 2.6, 3.8],
}


def look_at_camera_to_world(position, yaw_deg, pitch_deg):
    """
    Build a camera-to-world matrix in OpenCV convention (x right, y down, z forward).
    yaw_deg: heading in the world xy-plane, 0 = looking along +x, positive = turning left (towards +y).
    pitch_deg: positive = looking down.
    """
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    camera_to_world = np.eye(4)
    camera_to_world[:3, 0] = right
    camera_to_world[:3, 1] = down
    camera_to_world[:3, 2] = forward
    camera_to_world[:3, 3] = position
    return camera_to_world


def build_road(num_lanes, lane_width, median_width, x_min, x_max, step=2.0):
    """
    Two carriageways separated by a median. Returns (lanelines, road_boundaries, lane_centers).
    lane_centers: list of (y_center, direction) where direction = +1 drives towards +x, -1 towards -x.
    Right-hand traffic: +x traffic is on the -y side.
    """
    xs = np.arange(x_min, x_max + step, step)

    def polyline(y):
        return np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], axis=1).tolist()

    lanelines, road_boundaries, lane_centers = [], [], []
    for side, direction in [(-1, 1), (1, -1)]:
        inner = side * median_width / 2
        for i in range(num_lanes + 1):
            lanelines.append(polyline(inner + side * i * lane_width))
        for i in range(num_lanes):
            lane_centers.append((inner + side * (i + 0.5) * lane_width, direction))
        road_boundaries.append(polyline(inner - side * 0.5))
        road_boundaries.append(polyline(inner + side * (num_lanes * lane_width + 0.5)))

    return lanelines, road_boundaries, lane_centers


def sample_size(object_type, rng):
    """Vary vehicle sizes a bit: sedans / SUVs for cars, box trucks / semi-trailers for trucks."""
    if object_type == "Truck":
        length = rng.choice([8.0, 12.0, 16.5]) + rng.uniform(-0.5, 0.5)
        return [length, rng.uniform(2.4, 2.6), rng.uniform(3.2, 4.0)]
    length, width, height = OBJECT_SIZES["Car"]
    return [length + rng.uniform(-0.5, 0.6), width + rng.uniform(-0.1, 0.1), height + rng.uniform(-0.2, 0.3)]


def spawn_vehicles(lane_centers, num_frames, x_min, x_max, min_gap, max_gap, min_speed, max_speed, truck_ratio, rng):
    """
    Fill every lane with constant-speed vehicles for the whole clip. min_gap / max_gap are bumper-to-bumper
    distances in meters, so small values give dense traffic. Everyone in a lane drives at the same speed,
    so vehicles never overlap. Trucks keep out of the innermost (fastest) lane when there are 3+ lanes.
    Returns a list of vehicle dicts.
    """
    duration = num_frames / FPS
    lanes_per_side = len(lane_centers) // 2
    vehicles = []
    for lane_idx, (y, direction) in enumerate(lane_centers):
        # lane_rank 0 is next to the median (fast lane), outer lanes are slower
        lane_rank = lane_idx % lanes_per_side
        speed = rng.uniform(min_speed, max_speed) * (1.0 + 0.08 * (lanes_per_side - 1 - lane_rank))
        lane_truck_ratio = 0.0 if (lanes_per_side >= 3 and lane_rank == 0) else truck_ratio

        # cover from far upstream (vehicles that will enter during the clip) to the end of the road
        upstream = speed * duration
        entry = x_min if direction > 0 else x_max
        offset = -upstream + rng.uniform(0, max_gap)  # distance downstream of the entry point, front bumper
        while offset < (x_max - x_min):
            object_type = "Truck" if rng.random() < lane_truck_ratio else "Car"
            lwh = sample_size(object_type, rng)
            center = offset - lwh[0] / 2
            vehicles.append({
                "type": object_type,
                "lwh": lwh,
                "y": y + rng.normal(0, 0.15),
                "x0": entry + direction * center,
                "velocity": direction * speed,
                "yaw": 0.0 if direction > 0 else np.pi,
            })
            # next vehicle is further downstream: this one's length + a gap
            offset += lwh[0] + rng.uniform(min_gap, max_gap)
    return vehicles


def object_to_world(x, y, z, yaw):
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform[:3, 3] = [x, y, z]
    return transform


def hdmap_sample(clip_id, name, polylines):
    return {
        '__key__': clip_id,
        f'{name}.json': {
            'labels': [{'labelData': {'shape3d': {'polyline3d': {'vertices': p}}}} for p in polylines]
        },
    }


@click.command()
@click.option("--output_root", "-o", type=str, required=True, help="output folder in RDS-HQ format")
@click.option("--clip_id", "-c", type=str, default="highway_fixed_cam_000", help="clip id")
@click.option("--num_frames", "-n", type=int, default=121, help="number of frames at 30 fps (121 = one Cosmos chunk)")
@click.option("--num_lanes", type=int, default=3, help="lanes per direction")
@click.option("--lane_width", type=float, default=3.6, help="lane width in meters")
@click.option("--median_width", type=float, default=2.0, help="median width in meters")
@click.option("--cam_x", type=float, default=0.0, help="camera position along the highway (m)")
@click.option("--cam_y", type=float, default=-16.0, help="camera lateral position (m), negative = right shoulder")
@click.option("--cam_height", type=float, default=8.0, help="camera height above the road (m)")
@click.option("--yaw", type=float, default=15.0, help="camera heading in degrees, 0 = looking down the highway (+x)")
@click.option("--pitch", type=float, default=12.0, help="camera downward tilt in degrees")
@click.option("--hfov", type=float, default=60.0, help="horizontal field of view in degrees")
@click.option("--width", type=int, default=1280, help="image width")
@click.option("--height", type=int, default=720, help="image height")
@click.option("--min_gap", type=float, default=25.0, help="min bumper-to-bumper gap in meters (smaller = denser)")
@click.option("--max_gap", type=float, default=60.0, help="max bumper-to-bumper gap in meters")
@click.option("--min_speed", type=float, default=22.0, help="min lane speed in m/s (outer lanes)")
@click.option("--max_speed", type=float, default=28.0, help="max lane speed in m/s")
@click.option("--truck_ratio", type=float, default=0.15, help="fraction of vehicles that are trucks (outside the fast lane)")
@click.option("--seed", type=int, default=0, help="random seed for traffic")
def main(output_root, clip_id, num_frames, num_lanes, lane_width, median_width, cam_x, cam_y, cam_height,
         yaw, pitch, hfov, width, height, min_gap, max_gap, min_speed, max_speed, truck_ratio, seed):
    rng = np.random.default_rng(seed)
    x_min, x_max = cam_x - 300.0, cam_x + 700.0

    # 1. camera: same pose for every frame
    camera_to_world = look_at_camera_to_world(np.array([cam_x, cam_y, cam_height]), yaw, pitch)
    pose_sample = {'__key__': clip_id}
    for frame_idx in range(num_frames):
        pose_sample[f"{frame_idx:06d}.pose.{CAMERA_NAME}.npy"] = camera_to_world.astype(np.float32)
    write_to_tar(pose_sample, f"{output_root}/pose/{clip_id}.tar")

    fx = width / 2 / np.tan(np.deg2rad(hfov) / 2)
    intrinsic_sample = {
        '__key__': clip_id,
        f"pinhole_intrinsic.{CAMERA_NAME}.npy": np.array([fx, fx, width / 2, height / 2, width, height]),
    }
    write_to_tar(intrinsic_sample, f"{output_root}/pinhole_intrinsic/{clip_id}.tar")

    # 2. static map
    lanelines, road_boundaries, lane_centers = build_road(num_lanes, lane_width, median_width, x_min, x_max)
    write_to_tar(hdmap_sample(clip_id, 'lanelines', lanelines), f"{output_root}/3d_lanelines/{clip_id}.tar")
    write_to_tar(hdmap_sample(clip_id, 'road_boundaries', road_boundaries), f"{output_root}/3d_road_boundaries/{clip_id}.tar")

    # 3. moving vehicles, one entry per frame
    vehicles = spawn_vehicles(lane_centers, num_frames, x_min, x_max, min_gap, max_gap, min_speed, max_speed, truck_ratio, rng)
    object_sample = {'__key__': clip_id}
    for frame_idx in range(num_frames):
        t = frame_idx / FPS
        frame_objects = {}
        for track_id, v in enumerate(vehicles):
            x = v["x0"] + v["velocity"] * t
            if not (x_min <= x <= x_max):
                continue
            frame_objects[f"{track_id:04d}"] = {
                'object_to_world': object_to_world(x, v["y"], v["lwh"][2] / 2, v["yaw"]).tolist(),
                'object_lwh': v["lwh"],
                'object_is_moving': True,
                'object_type': v["type"],
            }
        object_sample[f"{frame_idx:06d}.all_object_info.json"] = frame_objects
    write_to_tar(object_sample, f"{output_root}/all_object_info/{clip_id}.tar")


if __name__ == "__main__":
    main()
