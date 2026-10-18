"""명령 카드 세션, 보낸 문장을 작업 문장으로 정책 실행"""
import logging
import threading
import time

from rig import Busy

log = logging.getLogger("metarobo")


class Command:
    """체크포인트 하나를 띄워 두고 명령마다 정책을 다시 실행"""
    def __init__(self, rig):
        self.rig = rig
        self.policy = None
        self.checkpoint = None  # 불러오는 중이거나 불러온 체크포인트
        self.label = ""
        self.loading = False
        self.error = ""
        self.messages = []  # {"text", "state": running/done/stopped/replaced, "t", "duration_s"}

    def load(self, checkpoint, label):
        """정책 로드(백그라운드), 이전 모델과 대화 기록은 정리"""
        if self.loading:
            raise Busy("모델을 읽는 중입니다")
        if self.rig.mode not in ("idle", "teleop"):
            raise Busy("조작 화면에서 장치를 먼저 연결하세요")
        self.end()
        self.loading, self.error, self.checkpoint, self.label = True, "", checkpoint, label
        log.info("명령 모델 로드 시작: %s", checkpoint)

        def load():
            """정책 로드"""
            t0 = time.time()
            try:
                self.policy = self.rig.load_policy(checkpoint)
                log.info("명령 모델 로드 완료 (%.1f초)", time.time() - t0)
            except Exception as e:
                log.exception("명령 모델 로드 실패")
                self.error = f"모델을 읽지 못했습니다: {e}"
                self.checkpoint = None
            finally:
                self.loading = False
        threading.Thread(target=load, daemon=True).start()

    def _running(self):
        """이 세션의 정책이 지금 로봇을 움직이는지"""
        return self.policy is not None and self.rig.mode == "policy" and self.rig.policy is self.policy

    def _settle(self, state="stopped"):
        """끝난 명령의 상태 확정, 시간 초과면 done, 그 외는 state (stopped, 새 명령으로 바뀌면 replaced)"""
        m = self.messages[-1] if self.messages else None
        if not m or m["state"] != "running" or self._running():
            return
        info = self.rig.policy_info or {}
        m["state"] = "done" if info.get("timed_out") else state
        m["duration_s"] = round(time.time() - m["t"], 1)

    def send(self, text):
        """명령 실행, 실행 중인 명령은 자세를 유지한 채 멈추고 새 문장으로 교체"""
        text = text.strip()
        if not text:
            raise ValueError("명령이 비어 있습니다")
        if not self.policy:
            raise Busy("명령을 실행할 모델을 먼저 불러오세요")
        if self.rig.rec:
            raise Busy("녹화 중에는 명령을 실행할 수 없습니다")
        if self._running():
            self.rig.stop_policy(go_home=False)
        self._settle("replaced")
        self.rig.run_policy(self.policy, text)
        self.messages.append({"text": text, "state": "running", "t": time.time(), "duration_s": None})
        log.info('명령 실행: "%s"', text)

    def stop(self, go_home=False):
        """실행 중인 명령 중지, go_home이면 시작 자세로 복귀"""
        if self._running() or (go_home and self.rig.mode == "idle"):
            self.rig.stop_policy(go_home=go_home)
        self._settle()

    def end(self):
        """모델 내리기, 실행 중이면 자세 유지한 채 중지"""
        if self._running():
            self.rig.stop_policy(go_home=False)
        self._settle()
        if self.checkpoint and not self.loading:
            log.info("명령 모델 내림 (명령 %d개)", len(self.messages))
        self.policy, self.messages = None, []
        if not self.loading:
            self.checkpoint, self.label = None, ""

    def in_use(self):
        """불러오는 중이거나 불러온 체크포인트 경로"""
        return self.checkpoint

    def status(self):
        """화면용 명령 상태와 대화 기록"""
        self._settle()
        p = self.rig.policy_info if self._running() else None
        return {
            "checkpoint": self.checkpoint, "label": self.label, "loaded": self.policy is not None,
            "loading": self.loading, "error": self.error, "running": self._running(),
            "elapsed_s": time.time() - p["t0"] if p else None, "steps": p["steps"] if p else None,
            "messages": [{k: v for k, v in m.items() if k != "t"} for m in self.messages],
        }
