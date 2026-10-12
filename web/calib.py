"""LeRobot의 calibrate()가 input()으로 Enter를 기다려서 같은 순서를 화면 버튼으로 분리"""
import math
import threading
import time
from datetime import datetime

import config as C
from rig import JOINTS, Busy

FULL_TURN = "wrist_roll"
NARROW_DEG = 90  # 이보다 좁게 기록된 관절은 끝까지 움직이지 않은 것으로 판단


def ticks_to_deg(v):
    """서보 틱(0~4095)을 중간 자세 기준 각도로 변환"""
    return (v - 2047) * 360 / 4096


class Calibration:
    """캘리브레이션 진행 상태와 서보 버스"""
    def __init__(self, rig):
        self.rig = rig
        self.arm = None
        self.dev = None
        self.step = 0  # 0 대기, 1 중간 자세, 2 범위 기록, 3 저장됨
        self.pos = {}
        self.mins = {}
        self.maxes = {}
        self.homings = None
        self.error = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _make(self, arm):
        """팔 종류별 LeRobot 장치 객체 생성 (포트는 열지 않음)"""
        if arm == "follower":
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            return SO101Follower(SO101FollowerConfig(port=C.FOLLOWER_PORT, id=C.FOLLOWER_ID))
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        return SO101Leader(SO101LeaderConfig(port=C.LEADER_PORT, id=C.LEADER_ID))

    def info(self, arm):
        """포트, 설정 이름, 마지막 캘리브레이션 시각"""
        port = C.FOLLOWER_PORT if arm == "follower" else C.LEADER_PORT
        name = C.FOLLOWER_ID if arm == "follower" else C.LEADER_ID
        last = None
        try:
            f = self._make(arm).calibration_fpath
            if f.is_file():
                last = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        return {"port": port, "id": name, "last": last}

    def start(self, arm):
        """1단계 시작: 버스 연결, 토크 해제, 위치 읽기 스레드 시작"""
        if self.rig.mode != "off":
            raise Busy("캘리브레이션 전에 조작 화면에서 장치 연결을 해제하세요")
        self.cancel()
        self.arm, self.error = arm, ""
        self.pos, self.mins, self.maxes, self.homings = {}, {}, {}, None
        if not self.rig.mock:
            dev = self._make(arm)
            dev.bus.connect()
            dev.bus.disable_torque()
            from lerobot.motors.feetech import OperatingMode
            for m in dev.bus.motors:
                dev.bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
            self.dev = dev
        self.step = 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read(self):
        """현재 서보 위치(틱), 모의 모드는 합성값"""
        if self.rig.mock:
            t = time.time()
            return {j: 2047 + int(1500 * math.sin(t * 0.8 + i)) for i, j in enumerate(JOINTS)}
        return self.dev.bus.sync_read("Present_Position", JOINTS, normalize=False, num_retry=5)

    def _read_loop(self):
        """위치 주기 읽기, 2단계에서는 최소/최대 갱신"""
        while not self._stop.is_set():
            try:
                with self._lock:
                    pos = self._read()
                    self.pos = pos
                    if self.step == 2:
                        for j, v in pos.items():
                            self.mins[j] = min(self.mins.get(j, v), v)
                            self.maxes[j] = max(self.maxes.get(j, v), v)
            except Exception as e:
                self.error = str(e)
            time.sleep(0.03)

    def middle(self):
        """1단계 완료: 현재 자세를 중간(2047)으로 홈 오프셋 기록"""
        if self.step != 1:
            raise Busy("1단계에서만 누를 수 있습니다")
        with self._lock:
            self.homings = {j: 0 for j in JOINTS} if self.rig.mock else self.dev.bus.set_half_turn_homings()
            pos = self._read()
            self.mins, self.maxes = dict(pos), dict(pos)
            self.step = 2

    def finish(self):
        """2단계 완료: 범위 검사 후 모터와 캘리브레이션 파일에 저장"""
        if self.step != 2:
            raise Busy("2단계에서만 누를 수 있습니다")
        same = [j for j in JOINTS if j != FULL_TURN and self.mins.get(j) == self.maxes.get(j)]
        if same:
            raise Busy(f"움직이지 않은 관절이 있습니다: {', '.join(same)}")
        with self._lock:
            if not self.rig.mock:
                from lerobot.motors import MotorCalibration
                cal = {}
                for j, m in self.dev.bus.motors.items():
                    lo, hi = (0, 4095) if j == FULL_TURN else (self.mins[j], self.maxes[j])
                    cal[j] = MotorCalibration(id=m.id, drive_mode=0, homing_offset=self.homings[j],
                                              range_min=lo, range_max=hi)
                self.dev.bus.write_calibration(cal)
                self.dev.calibration = cal
                self.dev._save_calibration()
            self.step = 3
        self._close()

    def cancel(self):
        """진행 중인 캘리브레이션 취소"""
        self._close()
        self.step = 0

    def _close(self):
        """읽기 스레드 종료와 버스 해제"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self.dev is not None:
            try:
                self.dev.bus.disconnect(disable_torque=True)
            except Exception:
                pass
            self.dev = None

    def status(self):
        """관절별 위치와 범위를 각도로 바꾼 화면용 상태"""
        joints = []
        for i, j in enumerate(JOINTS):
            row = {"name": j, "id": i + 1, "pos": None, "min": None, "max": None, "state": "대기"}
            if self.step >= 2 and j in self.pos:
                row["pos"] = ticks_to_deg(self.pos[j])
            if j == FULL_TURN and self.step >= 2:
                row["state"] = "전체 회전"
            elif self.step >= 2 and j in self.mins:
                row["min"], row["max"] = ticks_to_deg(self.mins[j]), ticks_to_deg(self.maxes[j])
                row["state"] = "범위 좁음" if row["max"] - row["min"] < NARROW_DEG else "기록됨"
            joints.append(row)
        return {"arm": self.arm, "step": self.step, "error": self.error, "joints": joints}
