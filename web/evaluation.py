"""실물 평가 세션"""
import logging
import threading
import time

import config as C
import store
from rig import Busy

log = logging.getLogger("metarobo")


class Evaluation:
    """체크포인트 하나로 물체 위치 격자를 차례로 평가"""
    def __init__(self, rig):
        self.rig = rig
        self.session = None
        self.policy = None
        self.loading = False
        self.error = ""
        self.trial_t0 = None
        self.checkpoint = None  # 불러오는 중이거나 평가 중인 체크포인트

    def prepare(self, checkpoint, label, task):
        """정책 로드(백그라운드) 후 평가 세션 생성"""
        if self.loading:
            raise Busy("모델을 읽는 중입니다")
        if self.rig.mode not in ("idle", "teleop"):
            raise Busy("조작 화면에서 장치를 먼저 연결하세요")
        self.end()
        self.loading, self.error, self.checkpoint = True, "", checkpoint
        log.info("평가 모델 로드 시작: %s", checkpoint)

        def load():
            """정책 로드와 세션 생성"""
            t0 = time.time()
            try:
                self.policy = self.rig.load_policy(checkpoint)
                self.session = store.new_eval(checkpoint, label, task)
                log.info("평가 모델 로드 완료 (%.1f초)", time.time() - t0)
            except Exception as e:
                log.exception("평가 모델 로드 실패")
                self.error = f"모델을 읽지 못했습니다: {e}"
            finally:
                self.loading = False
        threading.Thread(target=load, daemon=True).start()

    def _need_session(self):
        """평가 세션이 없으면 거부"""
        if not self.session:
            raise Busy("평가할 모델을 먼저 고르세요")

    def start_trial(self):
        """다음 위치에서 회차 시작, 정책 실행"""
        self._need_session()
        if self.trial_t0 is not None:
            raise Busy("이번 회차를 먼저 성공/실패로 정리하세요")
        if len(self.session["trials"]) >= C.EVAL_GRID[0] * C.EVAL_GRID[1]:
            raise Busy("모든 위치를 평가했습니다")
        self.rig.run_policy(self.policy, self.session["task"])
        self.trial_t0 = time.time()
        log.info("평가 회차 %d/%d 시작", len(self.session["trials"]) + 1, C.EVAL_GRID[0] * C.EVAL_GRID[1])

    def finish_trial(self, success, reason):
        """회차 결과 기록 후 시작 자세로 복귀"""
        self._need_session()
        if self.trial_t0 is None:
            raise Busy("진행 중인 회차가 없습니다")
        duration = time.time() - self.trial_t0
        self.trial_t0 = None
        self.rig.stop_policy(go_home=True)
        trial = {"pos": len(self.session["trials"]), "success": success, "reason": reason.strip(),
                 "duration_s": round(duration, 1)}
        log.info("평가 회차 %d 결과: %s (%.1f초)%s", trial["pos"] + 1, "성공" if success else "실패",
                 duration, f", {trial['reason']}" if trial["reason"] else "")
        self.session = store.add_trial(self.session["id"], trial)

    def end(self):
        """평가 세션 종료, 실행 중인 정책 중지"""
        if self.trial_t0 is not None and self.rig.mode == "policy":
            self.rig.stop_policy(go_home=False)
        if self.session:
            log.info("평가 종료 (%d회차 기록)", len(self.session["trials"]))
        self.session, self.policy, self.trial_t0 = None, None, None
        if not self.loading:
            self.checkpoint = None

    def in_use(self):
        """모델을 불러오는 중이거나 평가 중인 체크포인트 경로"""
        return self.checkpoint if self.loading or self.session else None

    def status(self):
        """화면용 평가 상태와 전체 기록"""
        return {
            "session": self.session, "loading": self.loading, "error": self.error,
            "trial_running": self.trial_t0 is not None, "grid": list(C.EVAL_GRID),
            "history": store.evals(),
        }
