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
    3d_poles/<clip_id>.tar           static roadside light poles and sign posts
    3d_traffic_signs/<clip_id>.tar   static road signs (cuboids)
    captions/<clip_id>.json          text prompts for Cosmos-Transfer, one per weather / time-of-day variation

World frame: x along the highway, y to the left, z up. The road surface is z = 0.
"""
import json
from pathlib import Path

import click
import numpy as np

from utils.wds_utils import write_to_tar

CAMERA_NAME = "roadside_cam"
FPS = 30

# length, width, height in meters
# Vehicle subtypes. 'lwh' is the standard size (length, width, height in meters), randomized by
# --size_variation. 'object_type' is the class the renderer understands (see config/hdmap_color_config.json),
# the fine-grained subtype is stored in 'object_subtype'. 'heavy' vehicles stay out of the fast lane.
VEHICLE_CLASSES = {
    "car":         {"lwh": [4.5, 1.8, 1.5],   "object_type": "Car",     "heavy": False},
    "pickup":      {"lwh": [5.5, 2.0, 1.9],   "object_type": "Car",     "heavy": False},
    "motorcycle":  {"lwh": [2.2, 0.8, 1.5],   "object_type": "Cyclist", "heavy": False},
    "truck":       {"lwh": [9.0, 2.5, 3.5],   "object_type": "Truck",   "heavy": True},
    "semi_truck":  {"lwh": [16.5, 2.55, 4.0], "object_type": "Truck",   "heavy": True},
    # car towing a trailer: two boxes, the car ('car') and the trailer ('trailer') behind it
    "car_trailer": {"lwh": [4.7, 1.85, 1.6],  "object_type": "Car",     "heavy": True},
}
TRAILER = {"lwh": [4.0, 2.0, 1.9], "object_type": "Trailer"}  # the renderer draws 'Trailer' as a Truck
HITCH_GAP = 1.0  # meters between the car's rear bumper and the trailer's front

DEFAULT_MIX = "car=0.45,pickup=0.15,motorcycle=0.08,truck=0.10,semi_truck=0.12,car_trailer=0.10"


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


def build_road(num_lanes, lane_width, median_width, x_min, x_max, shoulder_width=0.5, step=2.0):
    """
    Two carriageways separated by a median, with a paved shoulder of shoulder_width meters outside the last lane.
    Returns (lanelines, road_boundaries, lane_centers).
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
        road_boundaries.append(polyline(inner + side * (num_lanes * lane_width + shoulder_width)))

    return lanelines, road_boundaries, lane_centers


def parked_camera(side, facing, num_lanes, lane_width, median_width, shoulder_width):
    """
    Camera of a car parked on the outer shoulder, like an ego vehicle's front camera but standing still.
    side: 'right' (the -y edge, next to the +x carriageway) or 'left' (the +y edge, next to the -x carriageway).
    facing: 'with_traffic' looks the way the adjacent lanes drive (vehicles overtake and drive away),
            'against_traffic' looks at the adjacent lanes' oncoming vehicles (they approach and pass by).
    Returns (cam_y, yaw_deg).
    """
    sign = -1.0 if side == "right" else 1.0
    cam_y = sign * (median_width / 2 + num_lanes * lane_width + shoulder_width / 2)
    adjacent_direction = 1.0 if side == "right" else -1.0
    looks_along_x = adjacent_direction if facing == "with_traffic" else -adjacent_direction
    return cam_y, 0.0 if looks_along_x > 0 else 180.0


def parse_mix(mix):
    """'car=0.5,truck=0.2' -> {'car': 0.5, 'truck': 0.2}, normalized to sum to 1."""
    weights = {}
    for item in mix.split(","):
        name, weight = item.split("=")
        name = name.strip()
        if name not in VEHICLE_CLASSES:
            raise click.BadParameter(f"unknown vehicle '{name}', choose from {list(VEHICLE_CLASSES)}")
        weights[name] = float(weight)
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def random_size(standard_lwh, variation, rng):
    """Each dimension is drawn uniformly within +-variation (e.g. 0.2 = +-20%) of the standard size."""
    return (np.array(standard_lwh) * (1.0 + rng.uniform(-variation, variation, size=3))).tolist()


def spawn_vehicles(lane_centers, num_frames, x_min, x_max, min_gap, max_gap, min_speed, max_speed, mix,
                   size_variation, rng):
    """
    Fill every lane with constant-speed vehicles for the whole clip. min_gap / max_gap are bumper-to-bumper
    distances in meters, so small values give dense traffic. Everyone in a lane drives at the same speed,
    so vehicles never overlap. Heavy vehicles keep out of the innermost (fastest) lane when there are 3+ lanes.
    Returns a list of objects (a car with a trailer gives two objects).
    """
    duration = num_frames / FPS
    lanes_per_side = len(lane_centers) // 2
    light_mix = {k: w for k, w in mix.items() if not VEHICLE_CLASSES[k]["heavy"]} or mix
    objects = []
    for lane_idx, (y, direction) in enumerate(lane_centers):
        # lane_rank 0 is next to the median (fast lane), outer lanes are slower
        lane_rank = lane_idx % lanes_per_side
        speed = rng.uniform(min_speed, max_speed) * (1.0 + 0.08 * (lanes_per_side - 1 - lane_rank))
        lane_mix = light_mix if (lanes_per_side >= 3 and lane_rank == 0) else mix
        names, weights = list(lane_mix), np.array(list(lane_mix.values()))
        weights = weights / weights.sum()

        # cover from far upstream (vehicles that will enter during the clip) to the end of the road
        upstream = speed * duration
        entry = x_min if direction > 0 else x_max
        front_of_previous = -upstream  # distance downstream of the entry point
        while front_of_previous < (x_max - x_min):
            subtype = rng.choice(names, p=weights)
            vehicle_class = VEHICLE_CLASSES[subtype]
            # motorcycles wander more inside the lane
            lateral = y + rng.normal(0, 0.4 if subtype == "motorcycle" else 0.15)

            lwh = random_size(vehicle_class["lwh"], size_variation, rng)
            trailer_lwh = random_size(TRAILER["lwh"], size_variation, rng) if subtype == "car_trailer" else None
            total_length = lwh[0] + (HITCH_GAP + trailer_lwh[0] if trailer_lwh else 0.0)

            # this vehicle's rear (or its trailer's rear) is a gap ahead of the previous vehicle's front
            front = front_of_previous + rng.uniform(min_gap, max_gap) + total_length
            front_of_previous = front

            def add(object_type, object_subtype, size, object_front, extra=None):
                center = object_front - size[0] / 2
                objects.append({
                    "track_id": f"{len(objects):05d}",
                    "type": object_type,
                    "subtype": object_subtype,
                    "lwh": size,
                    "y": lateral,
                    "x0": entry + direction * center,
                    "velocity": direction * speed,
                    "yaw": 0.0 if direction > 0 else np.pi,
                    **(extra or {}),
                })

            add(vehicle_class["object_type"], subtype, lwh, front)
            if trailer_lwh:
                add(TRAILER["object_type"], "trailer", trailer_lwh, front - lwh[0] - HITCH_GAP,
                    {"towed_by": objects[-1]["track_id"]})
    return objects


def object_to_world(x, y, z, yaw):
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform[:3, 3] = [x, y, z]
    return transform


CAPTION_VARIATIONS = {
    "clear_day": "It is a bright sunny day with a clear blue sky, crisp shadows under the vehicles and dry asphalt.",
    "golden_hour": "It is late afternoon at golden hour; low warm sunlight casts long shadows across the lanes.",
    "night": "It is night; the highway is lit by orange sodium street lights, and vehicles show bright headlights "
             "and red tail lights reflecting on the road.",
    "rain": "It is raining heavily under a dark overcast sky; the road surface is wet and reflective, "
            "vehicles throw up spray, and raindrops streak past the camera.",
    "snow": "It is snowing; snow covers the roadside and median, the lanes are slushy with tire tracks, "
            "and visibility is reduced.",
    "fog": "Dense fog covers the highway; distant vehicles fade into the grey haze and only nearby vehicles are clear.",
}

VEHICLE_WORDS = {
    "car": "cars", "pickup": "pickup trucks", "motorcycle": "motorcycles", "truck": "box trucks",
    "semi_truck": "semi-trailer trucks", "car_trailer": "cars towing trailers",
}


def make_captions(num_lanes, cam_y, cam_height, yaw, median_width, lane_width, min_gap, max_gap, min_speed,
                  max_speed, mix):
    """Text prompts for Cosmos-Transfer, one per weather / time-of-day variation, describing the static camera view."""
    half_road = median_width / 2 + num_lanes * lane_width
    looks_along_x = np.cos(np.deg2rad(yaw)) > 0
    if abs(cam_y) > half_road and cam_height < 4.0:
        # car parked on the shoulder
        side = "right" if cam_y < 0 else "left"
        with_traffic = looks_along_x == (cam_y < 0)  # right-hand traffic: the -y carriageway drives towards +x
        view = ("vehicles in the nearest lanes overtake the parked car and drive away from the camera"
                if with_traffic else "vehicles in the nearest lanes approach head-on and pass close by the parked car")
        intro = (f"The video is captured by the front camera of a car parked on the {side} shoulder of a highway. "
                 f"The car is stationary, so the camera does not move, while traffic passes by: {view}.")
    else:
        if abs(cam_y) <= half_road:
            mount = f"on an overhead sign gantry about {cam_height:.0f} meters above the highway"
        else:
            mount = f"on a tall pole beside the highway, about {cam_height:.0f} meters above the road"
        facing = "looking along the direction of traffic" if looks_along_x else "looking towards oncoming traffic"
        intro = (f"The video is recorded by a static traffic surveillance camera mounted {mount}, {facing}. "
                 f"The camera does not move.")

    mean_gap, mean_speed = (min_gap + max_gap) / 2, (min_speed + max_speed) / 2
    if mean_speed < 10:
        traffic = "a traffic jam: vehicles are packed bumper to bumper and crawl slowly"
    elif mean_gap < 20:
        traffic = "heavy, dense traffic with short gaps between vehicles"
    elif mean_gap < 45:
        traffic = "moderate traffic flowing steadily"
    else:
        traffic = "light, free-flowing traffic at highway speed"

    common = [VEHICLE_WORDS[name] for name, weight in sorted(mix.items(), key=lambda kv: -kv[1]) if weight >= 0.05]
    vehicles = ", ".join(common[:-1]) + (f" and {common[-1]}" if len(common) > 1 else common[0] if common else "vehicles")

    base = (f"{intro} It shows a straight {2 * num_lanes}-lane divided highway with a central "
            f"median barrier, painted lane markings, street light poles and road signs along the roadside, "
            f"with {traffic}. The traffic includes {vehicles}. The camera is completely static: the road, the lane "
            f"markings, the poles and the signs stay fixed in the frame, and only the vehicles move.")
    return {name: f"{base} {weather}" for name, weather in CAPTION_VARIATIONS.items()}


def build_landmarks(num_lanes, lane_width, median_width, shoulder_width, x_min, x_max, pole_spacing, sign_spacing,
                    rng):
    """
    Static roadside landmarks: light poles along both road edges and in the median, and signs on posts.
    They never move, which gives the video model a clear cue that the camera is static.
    Returns (poles, traffic_signs): pole polylines [[bottom], [top]] and sign cuboids (8 vertices).
    """
    edge = median_width / 2 + num_lanes * lane_width + shoulder_width
    poles, signs = [], []
    if pole_spacing > 0:
        for y in (-(edge + 1.0), edge + 1.0, 0.0):  # both roadsides and the median
            for x in np.arange(x_min + rng.uniform(0, pole_spacing), x_max, pole_spacing):
                x = x + rng.uniform(-2.0, 2.0)
                poles.append([[x, y, 0.0], [x, y, rng.uniform(8.0, 11.0)]])
    if sign_spacing > 0:
        for side, facing in ((-1.0, -1.0), (1.0, 1.0)):  # signs face the traffic approaching on that side
            y = side * (edge + 1.5)
            for x in np.arange(x_min + rng.uniform(0, sign_spacing), x_max, sign_spacing):
                width, height, bottom = rng.uniform(0.8, 2.5), rng.uniform(0.6, 1.8), rng.uniform(2.0, 3.0)
                poles.append([[x, y, 0.0], [x, y, bottom + height]])  # the sign's post
                y0, y1, x1, z0, z1 = y - width / 2, y + width / 2, x - facing * 0.05, bottom, bottom + height
                # top face then bottom face, same corner order as the RDS-HQ traffic sign cuboids
                signs.append([[x, y1, z1], [x, y0, z1], [x1, y0, z1], [x1, y1, z1],
                              [x, y1, z0], [x, y0, z0], [x1, y0, z0], [x1, y1, z0]])
    return poles, signs


def hdmap_sample(clip_id, name, polylines, shape="polyline3d"):
    return {
        '__key__': clip_id,
        f'{name}.json': {
            'labels': [{'labelData': {'shape3d': {shape: {'vertices': p}}}} for p in polylines]
        },
    }


@click.command()
@click.option("--output_root", "-o", type=str, required=True, help="output folder in RDS-HQ format")
@click.option("--clip_id", "-c", type=str, default="highway_fixed_cam_000", help="clip id")
@click.option("--num_frames", "-n", type=int, default=121, help="number of frames at 30 fps (121 = one Cosmos chunk)")
@click.option("--num_lanes", type=int, default=3, help="lanes per direction")
@click.option("--lane_width", type=float, default=3.6, help="lane width in meters")
@click.option("--median_width", type=float, default=2.0, help="median width in meters")
@click.option("--shoulder_width", type=float, default=None,
              help="paved shoulder outside the last lane in meters (default 0.5, or 3.5 with --parked)")
@click.option("--parked", type=click.Choice(["right", "left"]), default=None,
              help="camera of a stationary car parked on the right / left shoulder (ego-car view); sets cam_y and yaw")
@click.option("--facing", type=click.Choice(["with_traffic", "against_traffic"]), default="with_traffic",
              help="with --parked: look the way the adjacent lanes drive, or towards their oncoming vehicles")
@click.option("--cam_x", type=float, default=0.0, help="camera position along the highway (m)")
@click.option("--cam_y", type=float, default=-16.0, help="camera lateral position (m), negative = right shoulder")
@click.option("--cam_height", type=float, default=None, help="camera height above the road (m) (default 8, or 1.5 with --parked)")
@click.option("--yaw", type=float, default=15.0, help="camera heading in degrees, 0 = looking down the highway (+x)")
@click.option("--pitch", type=float, default=None, help="camera downward tilt in degrees (default 12, or 1.5 with --parked)")
@click.option("--hfov", type=float, default=None, help="horizontal field of view in degrees (default 60, or 100 with --parked)")
@click.option("--width", type=int, default=1280, help="image width")
@click.option("--height", type=int, default=720, help="image height")
@click.option("--min_gap", type=float, default=25.0, help="min bumper-to-bumper gap in meters (smaller = denser)")
@click.option("--max_gap", type=float, default=60.0, help="max bumper-to-bumper gap in meters")
@click.option("--min_speed", type=float, default=22.0, help="min lane speed in m/s (outer lanes)")
@click.option("--max_speed", type=float, default=28.0, help="max lane speed in m/s")
@click.option("--mix", type=str, default=DEFAULT_MIX, show_default=True,
              help=f"vehicle mix as name=weight pairs, names: {', '.join(VEHICLE_CLASSES)}")
@click.option("--size_variation", type=float, default=0.2, help="random size variation per dimension (0.2 = +-20%)")
@click.option("--pole_spacing", type=float, default=40.0, help="meters between roadside light poles (0 = none)")
@click.option("--sign_spacing", type=float, default=150.0, help="meters between road signs on each side (0 = none)")
@click.option("--seed", type=int, default=0, help="random seed for traffic")
def main(parked, facing, **kwargs):
    defaults = {"shoulder_width": 3.5, "cam_height": 1.5, "pitch": 1.5, "hfov": 100.0} if parked else \
               {"shoulder_width": 0.5, "cam_height": 8.0, "pitch": 12.0, "hfov": 60.0}
    for name, value in defaults.items():
        if kwargs[name] is None:
            kwargs[name] = value
    if parked:
        kwargs["cam_y"], kwargs["yaw"] = parked_camera(parked, facing, kwargs["num_lanes"], kwargs["lane_width"],
                                                       kwargs["median_width"], kwargs["shoulder_width"])
    create_scene(**kwargs)


def create_scene(output_root, clip_id, num_frames=121, num_lanes=3, lane_width=3.6, median_width=2.0,
                 shoulder_width=0.5, cam_x=0.0, cam_y=-16.0, cam_height=8.0, yaw=15.0, pitch=12.0, hfov=60.0, width=1280, height=720,
                 min_gap=25.0, max_gap=60.0, min_speed=22.0, max_speed=28.0, mix=DEFAULT_MIX, size_variation=0.2,
                 pole_spacing=40.0, sign_spacing=150.0, seed=0):
    """Write one clip in RDS-HQ format. `mix` is a 'name=weight,...' string or a {name: weight} dict."""
    rng = np.random.default_rng(seed)
    if isinstance(mix, str):
        mix = parse_mix(mix)
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
    lanelines, road_boundaries, lane_centers = build_road(num_lanes, lane_width, median_width, x_min, x_max, shoulder_width)
    write_to_tar(hdmap_sample(clip_id, 'lanelines', lanelines), f"{output_root}/3d_lanelines/{clip_id}.tar")
    write_to_tar(hdmap_sample(clip_id, 'road_boundaries', road_boundaries), f"{output_root}/3d_road_boundaries/{clip_id}.tar")
    poles, signs = build_landmarks(num_lanes, lane_width, median_width, shoulder_width, x_min, x_max, pole_spacing,
                                   sign_spacing, np.random.default_rng(seed + 1))
    write_to_tar(hdmap_sample(clip_id, 'poles', poles), f"{output_root}/3d_poles/{clip_id}.tar")
    write_to_tar(hdmap_sample(clip_id, 'traffic_signs', signs, "cuboid3d"), f"{output_root}/3d_traffic_signs/{clip_id}.tar")

    # 3. moving vehicles, one entry per frame
    vehicles = spawn_vehicles(lane_centers, num_frames, x_min, x_max, min_gap, max_gap, min_speed, max_speed,
                              mix, size_variation, rng)
    object_sample = {'__key__': clip_id}
    for frame_idx in range(num_frames):
        t = frame_idx / FPS
        frame_objects = {}
        for v in vehicles:
            x = v["x0"] + v["velocity"] * t
            if not (x_min <= x <= x_max):
                continue
            frame_objects[v["track_id"]] = {
                'object_to_world': object_to_world(x, v["y"], v["lwh"][2] / 2, v["yaw"]).tolist(),
                'object_lwh': v["lwh"],
                'object_is_moving': True,
                'object_type': v["type"],
                'object_subtype': v["subtype"],
                'object_velocity': [v["velocity"], 0.0, 0.0],  # world frame, m/s
            }
            if "towed_by" in v:
                frame_objects[v["track_id"]]['towed_by'] = v["towed_by"]
        object_sample[f"{frame_idx:06d}.all_object_info.json"] = frame_objects
    write_to_tar(object_sample, f"{output_root}/all_object_info/{clip_id}.tar")

    # 4. text prompts for Cosmos-Transfer (one per weather / time-of-day variation)
    captions = make_captions(num_lanes, cam_y, cam_height, yaw, median_width, lane_width, min_gap, max_gap,
                             min_speed, max_speed, mix)
    caption_file = Path(output_root) / "captions" / f"{clip_id}.json"
    caption_file.parent.mkdir(parents=True, exist_ok=True)
    with open(caption_file, "w") as f:
        json.dump(captions, f, indent=4)


if __name__ == "__main__":
    main()
