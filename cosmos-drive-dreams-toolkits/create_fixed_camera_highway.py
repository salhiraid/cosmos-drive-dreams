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


def spawn_vehicles(lane_centers, num_frames, x_min, x_max, vehicles_per_lane, truck_ratio, rng):
    """Constant-speed vehicles per lane, spaced so they don't overlap. Returns a list of vehicle dicts."""
    duration = num_frames / FPS
    vehicles = []
    for lane_idx, (y, direction) in enumerate(lane_centers):
        # slower traffic on the outer lanes
        lane_rank = lane_idx % (len(lane_centers) // 2)
        speed = rng.uniform(22, 28) + 3.0 * (len(lane_centers) // 2 - 1 - lane_rank)

        # everyone in a lane drives at the same speed, so the gaps stay constant
        gaps = rng.uniform(25, 60, size=vehicles_per_lane)
        entry = x_min if direction > 0 else x_max
        offsets = np.cumsum(gaps) - rng.uniform(0, speed * duration)
        for offset in offsets:
            object_type = "Truck" if rng.random() < truck_ratio else "Car"
            vehicles.append({
                "type": object_type,
                "lwh": OBJECT_SIZES[object_type],
                "y": y + rng.normal(0, 0.15),
                "x0": entry + direction * offset,
                "velocity": direction * speed,
                "yaw": 0.0 if direction > 0 else np.pi,
            })
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
@click.option("--vehicles_per_lane", type=int, default=8, help="number of vehicles spawned per lane")
@click.option("--truck_ratio", type=float, default=0.15, help="fraction of vehicles that are trucks")
@click.option("--seed", type=int, default=0, help="random seed for traffic")
def main(output_root, clip_id, num_frames, num_lanes, lane_width, median_width, cam_x, cam_y, cam_height,
         yaw, pitch, hfov, width, height, vehicles_per_lane, truck_ratio, seed):
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
    vehicles = spawn_vehicles(lane_centers, num_frames, x_min, x_max, vehicles_per_lane, truck_ratio, rng)
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
