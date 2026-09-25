"""
Small highway traffic simulation used by create_fixed_camera_highway.py for lane changes and an ego vehicle.

Vehicles follow the car in front with the Intelligent Driver Model (IDM) and change lanes only into gaps that are
safe for both the new leader and the new follower. 'Zigzag' vehicles weave between neighbouring lanes every few
seconds, other vehicles change lanes now and then or when they are stuck behind a slower vehicle. A car towing a
trailer is simulated as one unit; the trailer follows the path of the car's rear. An optional ego vehicle drives
(or stands still) in one lane without changing lanes; it is not written as an object, the camera rides on it.
"""
import bisect

import numpy as np

FPS = 30
DT = 1.0 / FPS

# IDM parameters
MIN_GAP = 2.0          # jam distance (m)
TIME_HEADWAY = 0.8     # desired time gap to the leader (s)
COMFORT_DECEL = 2.5    # m/s^2
MAX_ACCEL = {"light": 1.8, "heavy": 1.0}

HEAVY_SUBTYPES = {"truck", "semi_truck", "car_trailer"}
EGO_LENGTH, EGO_WIDTH = 4.7, 1.9


class Unit:
    """One simulated vehicle, optionally with a trailer. s is the front bumper position along the driving direction."""

    def __init__(self, main, trailer, direction, lane, lateral_offset, lane_ys):
        self.main, self.trailer = main, trailer
        self.direction = direction
        self.lane_ys = lane_ys                  # lateral y of each lane of this carriageway, index 0 next to the median
        self.lane = lane
        self.offset = lateral_offset            # small lateral offset inside the lane
        self.main_length = main["lwh"][0] if main else EGO_LENGTH
        self.length = self.main_length + (trailer["hitch"] + trailer["lwh"][0] if trailer else 0.0)
        self.s = direction * main["x0"] + self.main_length / 2 if main else 0.0
        self.v = abs(main["velocity"]) if main else 0.0
        self.desired_speed = self.v
        self.heavy = bool(main) and main["subtype"] in HEAVY_SUBTYPES
        self.is_ego = main is None
        self.zigzag = False
        self.last_direction = 0                 # -1 / +1: side of the last lane change, zigzaggers alternate
        self.change = None                      # (from_lane, to_lane, start_time, duration) while changing lanes
        self.next_change_time = 0.0
        self.y = lane_ys[lane] + lateral_offset
        self.vy = 0.0
        self.history_s, self.history_y = [], []  # rear-bumper path, used to place the trailer

    @property
    def rear(self):
        return self.s - self.length

    def occupied_lanes(self):
        return {self.change[0], self.change[1]} if self.change else {self.lane}


def _idm_accel(unit, gap, leader_speed):
    if unit.desired_speed <= 0.1:
        return -COMFORT_DECEL * 2 if unit.v > 0 else 0.0
    a_max = MAX_ACCEL["heavy" if unit.heavy else "light"]
    free = 1.0 - (unit.v / unit.desired_speed) ** 4
    if gap is None:
        return a_max * free
    desired_gap = MIN_GAP + max(0.0, unit.v * TIME_HEADWAY +
                                unit.v * (unit.v - leader_speed) / (2 * np.sqrt(a_max * COMFORT_DECEL)))
    return a_max * (free - (desired_gap / max(gap, 0.1)) ** 2)


class LaneIndex:
    """Units of one carriageway sorted by front position in every lane they occupy (two lanes while changing)."""

    def __init__(self, units):
        self.lanes = {}
        for unit in units:
            for lane in unit.occupied_lanes():
                self.lanes.setdefault(lane, []).append(unit)
        for lane_units in self.lanes.values():
            lane_units.sort(key=lambda u: u.s)
        self.keys = {lane: [u.s for u in lane_units] for lane, lane_units in self.lanes.items()}

    def add(self, unit, lane):
        position = bisect.bisect_right(self.keys.setdefault(lane, []), unit.s)
        self.keys[lane].insert(position, unit.s)
        self.lanes.setdefault(lane, []).insert(position, unit)

    def neighbours(self, unit, lane):
        """Closest unit ahead (front further along) and behind in a lane, skipping the unit itself."""
        lane_units, keys = self.lanes.get(lane, []), self.keys.get(lane, [])
        position = bisect.bisect_right(keys, unit.s)
        ahead = next((u for u in lane_units[position:] if u is not unit), None)
        behind = next((u for u in reversed(lane_units[:position]) if u is not unit), None)
        return ahead, behind


def _lane_is_safe(unit, target, index):
    """Gap acceptance: normal drivers want comfortable gaps, zigzagging drivers squeeze into tight ones."""
    ahead, behind = index.neighbours(unit, target)
    front_gap, rear_gap, closing = (2.0, 0.12, 0.3) if unit.zigzag else (4.0, 0.4, 0.5)
    rear_min, rear_headway, rear_closing = (2.5, 0.2, 0.5) if unit.zigzag else (5.0, 0.6, 1.0)
    if ahead is not None and ahead.rear - unit.s < max(front_gap, rear_gap * unit.v +
                                                       closing * max(0.0, unit.v - ahead.v)):
        return False
    if behind is not None and unit.rear - behind.s < max(rear_min, rear_headway * behind.v +
                                                         rear_closing * max(0.0, behind.v - unit.v)):
        return False
    return True


def _allowed_lanes(unit, num_lanes):
    lanes = [unit.lane - 1, unit.lane + 1]
    lanes = [l for l in lanes if 0 <= l < num_lanes]
    if unit.heavy and num_lanes >= 3:
        lanes = [l for l in lanes if l != 0]   # heavy vehicles stay out of the fast lane
    return lanes


def simulate(vehicles, lane_centers, num_frames, warmup_frames, rng, zigzag_ratio=0.0, lane_change_rate=0.0,
             ego_lane=None, ego_x=0.0, ego_speed=0.0, zigzag_period=1.5, zigzag_near=0, view_x=0.0, view_dir=1.0):
    """
    zigzag_near: when the recording starts, the N light vehicles closest in front of the camera (10-150 m along
    view_dir from view_x, or from the ego car) also start zigzagging, so the weaving is always in view.
    zigzag_period: seconds per lane change of a zigzagging vehicle, pause included (1.0 = three lane changes in
    three seconds when the gaps allow it). Zigzaggers alternate left / right when they can.
    vehicles: output of spawn_vehicles (positions at the start of the warm-up).
    Returns (frames, ego_path):
        frames: list over recorded frames of {track_id: (x, y, yaw, vx, vy)} for every object (trailers included)
        ego_path: list over recorded frames of (x, y, yaw), or None without ego
    """
    by_direction = {}
    for y, direction in lane_centers:
        by_direction.setdefault(direction, []).append(y)
    for direction in by_direction:  # index 0 = next to the median (smallest |y|)
        by_direction[direction].sort(key=abs)

    trailers = {v["towed_by"]: v for v in vehicles if "towed_by" in v}
    units = {1: [], -1: []}
    for v in vehicles:
        if "towed_by" in v:
            continue
        direction = 1 if v["velocity"] > 0 else -1
        lane_ys = by_direction[direction]
        lane = int(np.argmin([abs(v["y"] - ly) for ly in lane_ys]))
        trailer = trailers.get(v["track_id"])
        if trailer is not None:
            trailer = dict(trailer, hitch=abs(v["x0"] - trailer["x0"]) - (v["lwh"][0] + trailer["lwh"][0]) / 2)
        unit = Unit(v, trailer, direction, lane, v["y"] - lane_ys[lane], lane_ys)
        can_zigzag = not unit.heavy
        unit.zigzag = can_zigzag and rng.random() < zigzag_ratio
        if unit.zigzag:
            unit.desired_speed *= rng.uniform(1.1, 1.3)  # weaving drivers are in a hurry
        unit.next_change_time = rng.uniform(0.0, 3.0)
        units[direction].append(unit)

    ego = None
    if ego_lane is not None:
        lane_ys = by_direction[1]
        ego = Unit(None, None, 1, ego_lane, 0.0, lane_ys)
        warmup = warmup_frames * DT
        ego.s = ego_x - ego_speed * warmup + EGO_LENGTH / 2
        ego.v = ego.desired_speed = ego_speed
        # clear the ego lane around the ego start position
        units[1] = [u for u in units[1] if not (u.lane == ego_lane and u.rear < ego.s + 25.0 and u.s > ego.rear - 20.0)]
        units[1].append(ego)

    frames, ego_path = [], []
    num_lanes = {d: len(ys) for d, ys in by_direction.items()}
    for step in range(warmup_frames + num_frames):
        t = step * DT
        if step == warmup_frames and zigzag_near > 0:
            origin, look = (ego.s, 1.0) if ego is not None else (view_x, view_dir)
            candidates = []
            for group in units.values():
                for unit in group:
                    if unit.is_ego or unit.heavy or unit.zigzag:
                        continue
                    ahead = (unit.direction * (unit.s - unit.main_length / 2) - origin) * look
                    if 10.0 <= ahead <= 150.0:
                        candidates.append((ahead, unit))
            for _, unit in sorted(candidates, key=lambda c: c[0])[:zigzag_near]:
                unit.zigzag = True
                unit.desired_speed *= rng.uniform(1.1, 1.3)
                unit.next_change_time = t + rng.uniform(0.0, 0.3)
        for direction, group in units.items():
            index = LaneIndex(group)
            # 1. lane change decisions
            for unit in group:
                if unit.is_ego or unit.change is not None:
                    continue
                targets = _allowed_lanes(unit, num_lanes[direction])
                if not targets:
                    continue
                want = False
                if unit.zigzag:
                    want = t >= unit.next_change_time
                else:
                    ahead, _ = index.neighbours(unit, unit.lane)
                    stuck = ahead is not None and unit.v < 0.7 * unit.desired_speed and ahead.rear - unit.s < 25.0
                    rate = lane_change_rate / 60.0 + (0.5 if stuck else 0.0)
                    want = rng.random() < rate * DT
                if not want:
                    continue
                rng.shuffle(targets)
                if unit.zigzag:  # zig then zag: prefer going back the way the last change came from
                    targets.sort(key=lambda lane: (lane - unit.lane) == unit.last_direction)
                for target in targets:
                    if _lane_is_safe(unit, target, index):
                        if unit.zigzag:
                            duration = rng.uniform(0.85, 1.0) * zigzag_period
                            pause = rng.uniform(0.0, 0.15) * zigzag_period
                        else:
                            duration, pause = rng.uniform(3.0, 5.0), 5.0
                        unit.change = (unit.lane, target, t, duration)
                        unit.last_direction = target - unit.lane
                        index.add(unit, target)
                        unit.next_change_time = t + duration + pause
                        break
                else:
                    unit.next_change_time = t + (0.1 if unit.zigzag else 0.3)  # retry soon

            # 2. longitudinal motion (IDM on the closest leader in any occupied lane)
            accels = []
            for unit in group:
                gap, leader_speed = None, 0.0
                for lane in unit.occupied_lanes():
                    ahead, _ = index.neighbours(unit, lane)
                    if ahead is not None and (gap is None or ahead.rear - unit.s < gap):
                        gap, leader_speed = ahead.rear - unit.s, ahead.v
                if unit.is_ego and ego_speed <= 0.1:
                    accels.append(None)
                else:
                    accels.append((_idm_accel(unit, gap, leader_speed), gap, leader_speed))
            for unit, accel in zip(group, accels):
                if accel is None:
                    unit.v = 0.0
                    continue
                a, gap, leader_speed = accel
                unit.v = max(0.0, unit.v + a * DT)
                if gap is not None and gap - unit.v * DT < 0.5:  # hard safety stop
                    unit.v = min(unit.v, leader_speed)
                unit.s += unit.v * DT

            # 3. lateral motion
            for unit in group:
                previous_y = unit.y
                if unit.change is not None:
                    source, target, start, duration = unit.change
                    u = min(1.0, (t - start) / duration)
                    blend = (1 - np.cos(np.pi * u)) / 2
                    unit.y = unit.lane_ys[source] + (unit.lane_ys[target] - unit.lane_ys[source]) * blend + unit.offset
                    if u >= 1.0:
                        unit.lane, unit.change = target, None
                else:
                    unit.y = unit.lane_ys[unit.lane] + unit.offset
                unit.vy = (unit.y - previous_y) / DT
                unit.history_s.append(unit.s - unit.main_length)
                unit.history_y.append(unit.y)

        if step < warmup_frames:
            continue

        frame = {}
        for direction, group in units.items():
            for unit in group:
                if unit.is_ego:
                    continue
                vx = direction * unit.v
                center_s = unit.s - unit.main_length / 2
                frame[unit.main["track_id"]] = (direction * center_s, unit.y, np.arctan2(unit.vy, vx), vx, unit.vy)
                if unit.trailer is not None:
                    trailer_s = unit.s - unit.main_length - unit.trailer["hitch"] - unit.trailer["lwh"][0] / 2
                    hs, hy = np.asarray(unit.history_s), np.asarray(unit.history_y)
                    hs = hs + np.arange(len(hs)) * 1e-6  # strictly increasing for interpolation
                    trailer_y = float(np.interp(trailer_s, hs, hy))
                    ahead_y = float(np.interp(trailer_s + 1.0, hs, hy))
                    trailer_yaw = np.arctan2(ahead_y - trailer_y, direction * 1.0)
                    frame[unit.trailer["track_id"]] = (direction * trailer_s, trailer_y, trailer_yaw, vx,
                                                       (ahead_y - trailer_y) * unit.v)
        frames.append(frame)
        if ego is not None:
            ego_path.append((ego.s - EGO_LENGTH / 2, ego.y, np.arctan2(ego.vy, max(ego.v, 1e-3))))

    return frames, (ego_path if ego is not None else None)
