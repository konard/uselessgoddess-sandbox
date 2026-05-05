"""Quick smoke test — exercises every module in `code/` and prints a
one-liner per subsystem so a reviewer can sanity-check them.

Run from repo root: python experiments/smoke_test.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.aim_trajectory import AimParams, fitts_movement_time, synthesize_aim
from code.counter_strafe import CSPhysics, CounterStrafer
from code.integration import CS2Bot, Perception
from code.navmesh_follow import NavMesh
from code.reaction_time import FlickRT, TrackingRT
from code.recoil_compensator import RecoilCompensator, RecoilParams, synthetic_ak47_table
from code.sensor_fusion import TargetEstimator


def test_fitts():
    mt = fitts_movement_time(20.0, 4.0, 0.10, 0.15)
    assert 0.2 < mt < 0.6, mt
    print(f"  fitts(20 deg, 4 deg) = {mt*1000:.1f} ms  (expect 200-600)")


def test_synth_aim():
    rng = np.random.default_rng(0)
    traj = synthesize_aim(np.array([0.0, 0.0]),
                          np.array([20.0, 4.0]),
                          AimParams(), rng=rng)
    assert traj.shape[1] == 2
    end = traj[-1]
    err = np.linalg.norm(end - np.array([20.0, 4.0]))
    print(f"  synth_aim end-error = {err:.3f} deg  ({len(traj)} ticks)")
    assert err < 0.6, err


def test_estimator():
    est = TargetEstimator()
    for k in range(8):
        est.predict()
        est.update_repvit(10.0 + 0.05 * k, 1.0)
    for k in range(8):
        est.predict()
        est.update_yolo(20.0 + 0.5 * k, 2.0, 0.85, 0.04)
    y, p = est.target()
    print(f"  estimator target = ({y:.2f}, {p:.2f})  (expect ~ (24, 2))")


def test_counter_strafe():
    cs = CounterStrafer(CSPhysics())
    keys_far = cs.tick(0.0, 200.0, 0.0, want_fire=True, want_dir=(1, 0))
    assert keys_far["a"], "should press A to counter v_x>0"
    print(f"  counter-strafe duration for v=200 = {cs.cs_duration(200)*1000:.1f} ms")


def test_recoil():
    table = synthetic_ak47_table(30)
    rc = RecoilCompensator(table, RecoilParams())
    rc.on_shot(0.0)
    rc.on_shot(0.10)
    rc.on_shot(0.20)
    dy, dp = rc.compensation(0.25)
    print(f"  recoil compensation @ shot3 = ({dy:.3f}, {dp:.3f}) deg")


def test_navmesh():
    # tiny mesh: 5 nodes in a line
    nodes = []
    for i in range(5):
        nodes.append({"id": i, "pos": [i * 100.0, 0.0, 0.0],
                      "neighbors": [i - 1, i + 1] if 0 < i < 4 else
                      ([1] if i == 0 else [3])})
    path = "/tmp/test_navmesh.json"
    json.dump({"nodes": nodes}, open(path, "w"))
    nm = NavMesh.from_json(path)
    p = nm.astar(0, 4)
    print(f"  navmesh A* path = {p}")
    assert p == [0, 1, 2, 3, 4]
    os.remove(path)


def test_rt():
    fl = FlickRT()
    samples = [fl.sample() for _ in range(2000)]
    med = sorted(samples)[len(samples) // 2]
    print(f"  flick RT median = {med*1000:.1f} ms  (expect 220-280)")
    assert 0.18 < med < 0.40


def test_full_loop():
    nodes = [{"id": 0, "pos": [0., 0., 0.], "neighbors": [1]},
             {"id": 1, "pos": [200., 0., 0.], "neighbors": [0, 2]},
             {"id": 2, "pos": [400., 100., 0.], "neighbors": [1]}]
    json.dump({"nodes": nodes}, open("/tmp/_nm.json", "w"))
    bot = CS2Bot(NavMesh.from_json("/tmp/_nm.json"))
    bot.follower.replan(np.array([0., 0., 0.]),
                        np.array([400., 100., 0.]))
    t0 = time.time()
    for k in range(64):
        perc = Perception(
            player_pos=np.array([k * 3.0, 0., 0.]),
            player_vel=np.array([192., 0., 0.]),
            cam_yaw=0.0, cam_pit=0.0,
            repvit_yaw=2.0 + 0.01 * k, repvit_pit=0.0,
            yolo_target=(5.0 + 0.05 * k, 0.5, 0.9, 0.05) if k > 8 else None,
        )
        out = bot.step(t0 + k / 64.0, perc)
    print(f"  CS2Bot loop ran 64 ticks; final yaw={out['yaw']:.2f}, "
          f"keys={[k for k,v in out['keys'].items() if v]}, fire={out['fire']}")
    os.remove("/tmp/_nm.json")


def main():
    print("[1] Fitts");          test_fitts()
    print("[2] synth_aim");      test_synth_aim()
    print("[3] estimator");      test_estimator()
    print("[4] counter-strafe"); test_counter_strafe()
    print("[5] recoil");         test_recoil()
    print("[6] navmesh A*");     test_navmesh()
    print("[7] reaction time");  test_rt()
    print("[8] full loop");      test_full_loop()
    print("\nALL OK")


if __name__ == "__main__":
    main()
