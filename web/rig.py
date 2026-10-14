"""시리얼 포트와 카메라는 한 프로세스만 열 수 있어 텔레옵, 녹화, 정책 실행을 모두 이 객체가 담당

데이터셋 쓰기는 제어 스레드 전용이라 다른 스레드는 call()로 작업 전달
"""
import logging
import math
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future

import cv2
import numpy as np

import config as C
import store

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HISTORY_S = 10
HOMING_S = 2.0
log = logging.getLogger("metarobo")


class Busy(Exception):
    """화면에 그대로 보여줄 요청 거부 사유"""


def mock_pose(t):
    """모의 리더 자세, 시간에 따른 관절별 사인파"""
    s = math.sin
    return {
        "shoulder_pan.pos": 35 * s(t * 0.5),
        "shoulder_lift.pos": -20 + 25 * s(t * 0.4 + 1),
        "elbow_flex.pos": 40 + 30 * s(t * 0.45 + 2),
        "wrist_flex.pos": 20 * s(t * 0.6),
        "wrist_roll.pos": 15 * s(t * 0.3),
        "gripper.pos": 50 + 45 * s(t * 0.7),
    }


class MockLeader:
    """모의 리더 암"""
    def get_action(self):
        return mock_pose(time.time())

    def disconnect(self):
        pass


class MockRobot:
    """모의 팔로워, 목표를 늦게 따라가는 관절과 합성 카메라 영상"""
    name = "so_follower"

    def __init__(self):
        self.pos = {f"{j}.pos": 0.0 for j in JOINTS}
        self.goal = dict(self.pos)
        self.observation_features = {**{k: float for k in self.pos}, **{c: (C.HEIGHT, C.WIDTH, 3) for c in C.CAMERAS}}
        self.action_features = {k: float for k in self.pos}

    def get_observation(self):
        for k in self.pos:
            self.pos[k] += 0.3 * (self.goal[k] - self.pos[k])
        obs = dict(self.pos)
        for name in C.CAMERAS:
            obs[name] = self._frame(name)
        return obs

    def _frame(self, name):
        """관절 위치를 원으로 그린 합성 카메라 프레임"""
        img = np.full((C.HEIGHT, C.WIDTH, 3), 236, np.uint8)
        x = int(C.WIDTH / 2 + self.pos["shoulder_pan.pos"] * 4)
        y = int(C.HEIGHT / 2 - self.pos["elbow_flex.pos"] * 2 + 60)
        r = int(20 + self.pos["gripper.pos"] / 3)
        cv2.circle(img, (x, y), r, (37, 99, 235), -1)
        cv2.putText(img, f"MOCK {name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (91, 100, 114), 2)
        return img

    def send_action(self, action):
        self.goal.update(action)
        return action

    def disconnect(self):
        pass


class Rig:
    """장치 세션, 연결부터 제어 루프, 녹화, 정책 실행까지 담당"""
    def __init__(self, mock=False):
        self.mock = mock
        self.mode = "off"  # off | connecting | idle | teleop | policy | homing
        self.error = ""
        self.robot = self.leader = None
        self.frames = {}
        self.hist = deque(maxlen=C.FPS * HISTORY_S)
        self.home = None
        self.dataset = None
        self.rec = None  # {"state": "recording"|"pending", "t0", "frames", "task"}
        self.last_saved = None
        self.policy = None
        self.policy_info = {}
        self._homing = None
        self._halt = False
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = None

    def connect(self):
        """제어 스레드 시작, 장치 열기는 스레드 안에서 진행"""
        if self.mode != "off":
            raise Busy("이미 연결되어 있습니다")
        self.error = ""
        self.mode = "connecting"
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rig", daemon=True)
        self._thread.start()

    def disconnect(self):
        """제어 스레드 종료 후 대기"""
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=30)

    def _open_devices(self):
        """팔로워, 리더, 카메라 열기, 캘리브레이션 없으면 중단"""
        if self.mock:
            self.robot, self.leader = MockRobot(), MockLeader()
            return
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        cams = {n: OpenCVCameraConfig(index_or_path=i, width=C.WIDTH, height=C.HEIGHT, fps=C.FPS)
                for n, i in C.CAMERAS.items()}
        robot = SO101Follower(SO101FollowerConfig(port=C.FOLLOWER_PORT, id=C.FOLLOWER_ID, cameras=cams))
        leader = SO101Leader(SO101LeaderConfig(port=C.LEADER_PORT, id=C.LEADER_ID))
        # calibrate=False: 캘리브레이션이 없으면 input() 대기 대신 화면에 알림
        leader.connect(calibrate=False)
        self.leader = leader
        robot.connect(calibrate=False)
        self.robot = robot
        missing = [n for n, d in (("팔로워", robot), ("리더", leader)) if not d.is_calibrated]
        if missing:
            raise Busy(f"{', '.join(missing)} 캘리브레이션이 필요합니다. 캘리브레이션 화면에서 먼저 진행하세요.")

    def _run(self):
        """제어 스레드 본체, 장치 연결 후 FPS 주기로 _tick 반복"""
        try:
            self._open_devices()
            obs = self.robot.get_observation()
            self.home = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in JOINTS}
            self.mode = "idle"
            period = 1 / C.FPS
            tick = 0
            while not self._stop.is_set():
                t0 = time.perf_counter()
                self._tick(tick, t0)
                tick += 1
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        except Exception as e:
            log.exception("장치 세션 오류")
            self.error = str(e) or type(e).__name__
        finally:
            self._close()

    def _close(self):
        """데이터셋 마무리, 장치 해제, 대기 중인 요청 실패 처리"""
        try:
            self._close_dataset()
        except Exception:
            log.exception("데이터셋 정리 실패")
        for dev in (self.robot, self.leader):
            try:
                if dev is not None and getattr(dev, "is_connected", True):
                    dev.disconnect()
            except Exception:
                log.exception("연결 해제 실패")
        self.robot = self.leader = None
        self.policy = None
        self.frames = {}
        while not self._q.empty():
            _, fut = self._q.get_nowait()
            fut.set_exception(Busy("장치 연결이 끊겼습니다"))
        self.mode = "off"

    def call(self, fn, timeout=60):
        """제어 스레드에서 fn 실행 후 결과 반환"""
        if self.mode in ("off", "connecting"):
            raise Busy("장치가 연결되어 있지 않습니다")
        fut = Future()
        self._q.put((fn, fut))
        return fut.result(timeout=timeout)

    def _tick(self, tick, t0):
        """제어 주기 1회: 요청 처리, 관측, 동작 결정과 전송, 녹화, 화면용 데이터 갱신"""
        # 다른 스레드가 넘긴 요청 처리
        while not self._q.empty():
            fn, fut = self._q.get_nowait()
            try:
                fut.set_result(fn())
            except Exception as e:
                fut.set_exception(e)

        obs = self.robot.get_observation()
        actual = {j: float(obs[f"{j}.pos"]) for j in JOINTS}

        # 비상 정지: 녹화와 정책 중단 후 현재 자세로 고정
        if self._halt:
            self._halt = False
            self._stop_recording(discard=True)
            self.policy = None
            self.mode = "idle"
            self.robot.send_action({f"{j}.pos": v for j, v in actual.items()})

        # 모드별 목표 동작 결정
        action, infer_ms = None, None
        if self.mode == "teleop":
            action = self.leader.get_action()
        elif self.mode == "policy":
            s = time.perf_counter()
            action = self.policy(obs)
            infer_ms = (time.perf_counter() - s) * 1000
            self.policy_info["steps"] += 1
            if time.time() - self.policy_info["t0"] >= C.EPISODE_TIME_S:
                self.policy_info["timed_out"] = True
                self.mode = "idle"
        elif self.mode == "homing":
            action = self._homing_step()
        if action is not None:
            self.robot.send_action(action)

        # 녹화 중이면 프레임 기록, 시간 초과 시 결과 대기로 전환
        if self.rec and self.rec["state"] == "recording" and action is not None:
            from lerobot.utils.constants import ACTION, OBS_STR
            from lerobot.utils.feature_utils import build_dataset_frame

            feats = self.dataset.features
            try:
                self.dataset.add_frame({**build_dataset_frame(feats, obs, prefix=OBS_STR),
                                        **build_dataset_frame(feats, action, prefix=ACTION),
                                        "task": self.rec["task"]})
            except Exception as e:
                log.exception("프레임 기록 실패")
                self.error = f"녹화를 멈췄습니다: {e}"
                self._stop_recording(discard=True)
                return
            self.rec["frames"] += 1
            if time.time() - self.rec["t0"] >= C.EPISODE_TIME_S:
                self.rec["state"] = "pending"

        # 화면 스트리밍용 JPEG 갱신 (2틱마다)
        if tick % 2 == 0:
            for name in C.CAMERAS:
                img = obs.get(name)
                if img is not None:
                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        self.frames[name] = buf.tobytes()

        # 그래프용 이력 기록
        self.hist.append({
            "t": time.time(),
            "actual": actual,
            "target": {j: float(action[f"{j}.pos"]) for j in JOINTS} if action else None,
            "load": self._gripper_load(actual, tick),
            "loop_ms": (time.perf_counter() - t0) * 1000,
            "infer_ms": infer_ms,
        })

    def _gripper_load(self, actual, tick):
        """그리퍼 부하(%), 버스 부담을 줄이려 3틱마다 읽기"""
        if self.mock:
            return 41.0 if actual["gripper"] < 25 else 4.0
        if tick % 3:
            return self.hist[-1]["load"] if self.hist else None
        try:
            # STS3215 Present_Load는 0.1% 단위, 부호는 방향
            return abs(self.robot.bus.sync_read("Present_Load", ["gripper"])["gripper"]) / 10
        except Exception:
            return None

    def _homing_step(self):
        """시작 자세까지 선형 보간한 목표, 끝나면 대기 모드"""
        h = self._homing
        a = min(1.0, (time.time() - h["t0"]) / HOMING_S)
        if a >= 1.0:
            self.mode = "idle"
        return {k: h["from"][k] + (h["to"][k] - h["from"][k]) * a for k in h["to"]}

    def estop(self):
        """정책, 텔레옵, 녹화를 멈추고 지금 자세 유지, 녹화 중이던 에피소드는 폐기"""
        if self.mode not in ("off", "connecting"):
            self._halt = True

    def set_teleop(self, on):
        """텔레옵 켜기/끄기, 끌 때 녹화 폐기 후 데이터셋 마무리"""
        def fn():
            if on:
                if self.mode not in ("idle", "teleop"):
                    raise Busy("정책 실행 중에는 텔레옵을 시작할 수 없습니다")
                self.mode = "teleop"
            else:
                self._stop_recording(discard=True)
                self._close_dataset()
                if self.mode == "teleop":
                    self.mode = "idle"
        self.call(fn)

    def record_start(self, task):
        """에피소드 녹화 시작, 데이터셋이 없으면 생성하고 있으면 이어 쓰기"""
        task = task.strip()
        if not task:
            raise Busy("작업 문장을 입력하세요")

        def fn():
            if self.mode not in ("idle", "teleop"):
                raise Busy("정책 실행 중에는 녹화할 수 없습니다")
            if self.rec:
                raise Busy("녹화 중인 에피소드를 먼저 성공/실패로 정리하세요")
            self._open_dataset()
            self.error = ""
            self.mode = "teleop"
            self.rec = {"state": "recording", "t0": time.time(), "frames": 0, "task": task}
        self.call(fn, timeout=120)

    def record_finish(self, result):
        """에피소드 저장 후 성공/실패 라벨 기록, result=None이면 폐기(재녹화)"""
        def fn():
            if not self.rec:
                raise Busy("녹화 중인 에피소드가 없습니다")
            if result is None:
                self._stop_recording(discard=True)
                return None
            if self.rec["frames"] == 0:
                self._stop_recording(discard=True)
                raise Busy("기록된 프레임이 없어 버렸습니다")
            idx = self.dataset.meta.total_episodes
            length = self.rec["frames"] / C.FPS
            self.dataset.save_episode()
            self.rec = None
            store.set_label(self.dataset.repo_id, idx, result=result)
            self.last_saved = {"index": idx, "result": result, "length_s": length}
            return idx
        return self.call(fn, timeout=300)

    def _open_dataset(self):
        """녹화용 데이터셋 열기, 기존 데이터셋은 resume"""
        if self.dataset is not None:
            return
        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import hw_to_dataset_features

        repo = store.record_repo(self.mock)
        # resume()은 root 필수, create()도 같은 폴더로 고정
        root = store.lerobot_home() / repo
        kw = dict(root=root, image_writer_threads=4 * len(C.CAMERAS), streaming_encoding=True, encoder_threads=2)
        if (root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset.resume(repo, **kw)
        else:
            features = {**hw_to_dataset_features(self.robot.action_features, ACTION),
                        **hw_to_dataset_features(self.robot.observation_features, OBS_STR)}
            self.dataset = LeRobotDataset.create(repo, C.FPS, features=features, robot_type=self.robot.name, **kw)

    def _stop_recording(self, discard):
        """녹화 상태 해제, discard면 에피소드 버퍼 폐기"""
        if self.rec and discard and self.dataset is not None:
            self.dataset.clear_episode_buffer()
        self.rec = None

    def _close_dataset(self):
        """녹화 정리 후 데이터셋 finalize"""
        self._stop_recording(discard=True)
        if self.dataset is not None:
            ds, self.dataset = self.dataset, None
            ds.finalize()

    def load_policy(self, path):
        """체크포인트로 obs -> action 함수 생성, 오래 걸려 제어 스레드 밖에서 호출"""
        if path == "mock":
            return lambda obs: mock_pose(time.time())
        if self.robot is None:
            raise Busy("장치가 연결되어 있지 않습니다")
        import torch
        from lerobot.common.control_utils import predict_action
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors
        from lerobot.policies.utils import make_robot_action
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

        path = str(store.ROOT / path)
        # CUDA 없으면 CPU 추론
        device = C.DEVICE if (C.DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
        pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path,
                                             preprocessor_overrides={"device_processor": {"device": device}})
        # 관측/동작을 데이터셋 형식으로 맞추는 feature 정의
        feats = {**hw_to_dataset_features(self.robot.observation_features, OBS_STR),
                 **hw_to_dataset_features(self.robot.action_features, ACTION)}
        robot_type = self.robot.name
        dev = torch.device(device)

        def run(obs):
            """관측 → 정책 추론 → 로봇 동작"""
            frame = build_dataset_frame(feats, obs, prefix=OBS_STR)
            act = predict_action(frame, policy, dev, pre, post, cfg.use_amp,
                                 task=self.policy_info.get("task"), robot_type=robot_type)
            return make_robot_action(act, feats)

        def reset():
            """회차 시작 전 정책과 전후처리 상태 초기화"""
            policy.reset()
            pre.reset()
            post.reset()
        run.reset = reset
        return run

    def run_policy(self, fn, task):
        """정책 실행 모드로 전환, 녹화 데이터셋은 닫기"""
        def go():
            if self.mode not in ("idle", "teleop"):
                raise Busy("다른 동작이 진행 중입니다")
            self._close_dataset()
            if hasattr(fn, "reset"):
                fn.reset()
            self.policy = fn
            self.policy_info = {"t0": time.time(), "steps": 0, "task": task, "timed_out": False}
            self.mode = "policy"
        self.call(go)

    def stop_policy(self, go_home=True):
        """정책 중지, go_home이면 시작 자세로 복귀"""
        def fn():
            self.policy = None
            if self.mode in ("policy", "idle", "homing"):
                if go_home and self.home and self.hist:
                    cur = {f"{j}.pos": v for j, v in self.hist[-1]["actual"].items()}
                    self._homing = {"t0": time.time(), "from": cur, "to": dict(self.home)}
                    self.mode = "homing"
                else:
                    self.mode = "idle"
        self.call(fn)

    def status(self):
        """화면용 세션 상태"""
        rec = None
        if self.rec:
            rec = {"state": self.rec["state"], "elapsed_s": time.time() - self.rec["t0"],
                   "frames": self.rec["frames"], "task": self.rec["task"]}
        pol = None
        if self.policy_info:
            pol = {"elapsed_s": time.time() - self.policy_info["t0"], "steps": self.policy_info["steps"],
                   "timed_out": self.policy_info["timed_out"], "running": self.mode == "policy"}
        return {
            "mode": self.mode, "mock": self.mock, "error": self.error,
            "cameras": sorted(self.frames), "recording": rec, "policy": pol,
            "dataset": store.record_repo(self.mock), "last_saved": self.last_saved,
            "episode_time_s": C.EPISODE_TIME_S,
        }

    def telemetry(self, joint):
        """관절 그래프용 최근 10초 이력, 약 150점 이하로 축소"""
        h = list(self.hist)
        if not h:
            return {"t": [], "actual": [], "target": [], "load": [], "loop_ms": [], "infer_ms": []}
        now = time.time()
        step = max(1, len(h) // 150)
        h = h[::step]
        return {
            "t": [round(x["t"] - now, 3) for x in h],
            "actual": [round(x["actual"][joint], 2) for x in h],
            "target": [round(x["target"][joint], 2) if x["target"] else None for x in h],
            "load": [x["load"] for x in h],
            "loop_ms": [round(x["loop_ms"], 1) for x in h],
            "infer_ms": [round(x["infer_ms"], 1) if x["infer_ms"] is not None else None for x in h],
        }
