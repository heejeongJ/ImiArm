"""metarobo 웹 화면 서버 (표준 라이브러리 http.server)"""
import json
import logging
import mimetypes
import platform
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import config as C
import store
from calib import Calibration
from command import Command
from evaluation import Evaluation
from rig import JOINTS, Busy, Rig
from training import Trainer, gpu_info

STATIC = Path(__file__).resolve().parent / "static"
PAGES = {"/", "/data", "/train", "/eval", "/sim", "/calib"}
log = logging.getLogger("metarobo")


class App:
    """화면 API 구현, 조회는 쿼리(q), 명령은 JSON 본문(b)"""
    def __init__(self, mock):
        self.rig = Rig(mock=mock)
        self.calib = Calibration(self.rig)
        self.trainer = Trainer()
        self.eval = Evaluation(self.rig)
        self.command = Command(self.rig)

    def overview(self, q):
        """장비 구성표와 화면 공통 설정"""
        gpu = gpu_info()
        cams = list(C.CAMERAS.items())
        return {
            "mock": self.rig.mock,
            "devices": [
                {"name": "SO-101 팔로워", "role": "정책 실행, 시연 기록", "port": C.FOLLOWER_PORT,
                 "spec": "STS3215 서보 6개", "setting": f"1Mbps, {C.FPS}Hz"},
                {"name": "SO-101 리더", "role": "원격 조작 입력", "port": C.LEADER_PORT,
                 "spec": "STS3215 서보 6개", "setting": f"1Mbps, {C.FPS}Hz"},
                *[{"name": f"카메라 {n}", "role": "데이터 키 observation.images." + n, "port": f"camera {i}",
                   "spec": "USB 카메라 (OpenCV)", "setting": f"{C.WIDTH}×{C.HEIGHT}, {C.FPS}프레임"} for n, i in cams],
                {"name": "연산 장치", "role": "정책 추론, 학습", "port": platform.node() or "이 컴퓨터",
                 "spec": gpu["name"] or "GPU 없음 (CPU)",
                 "setting": f"그래픽 메모리 {gpu['total_gb']:.0f}GB" if gpu["total_gb"] else f"설정 장치 {C.DEVICE}"},
            ],
            "cameras": [n for n, _ in cams], "joints": JOINTS, "fps": C.FPS,
            "task": C.TASK, "target_episodes": C.TARGET_EPISODES,
            "episode_time_s": C.EPISODE_TIME_S, "policy": C.POLICY,
            "train_defaults": {"steps": C.TRAIN_STEPS, "batch": C.BATCH_SIZE},
        }

    def rig_status(self, q):
        """장치 상태와 녹화 집계"""
        s = self.rig.status()
        repo = s["dataset"]
        lab = store.labels(repo)
        info = store._read(store.lerobot_home() / repo / "meta" / "info.json", {})
        lengths = [v["length_s"] for v in lab.values() if "length_s" in v]
        s["counts"] = {
            "total": info.get("total_episodes", 0),
            "success": sum(v.get("result") == "success" for v in lab.values()),
            "fail": sum(v.get("result") == "fail" for v in lab.values()),
            "avg_length_s": sum(lengths) / len(lengths) if lengths else None,
            "target": C.TARGET_EPISODES,
        }
        return s

    def telemetry(self, q):
        """관절 그래프 데이터"""
        return self.rig.telemetry(q.get("joint", "shoulder_pan"))

    def dataset(self, q):
        """데이터셋 에피소드 표"""
        return store.dataset_detail(q["repo"])

    def signal(self, q):
        """에피소드 그리퍼 시계열"""
        return store.episode_signal(q["repo"], int(q["ep"]), int(q["chunk"]), int(q["file"]))

    def train_status(self, q):
        """학습 상태, 체크포인트, 데이터셋 목록, policy를 주면 그 정책의 최근 학습"""
        return {**self.trainer.status(q.get("policy")),"checkpoints": store.checkpoints(), "runs": store.runs(),
                "datasets": store.list_datasets()}

    def train_run(self, q):
        """지난 학습 하나의 설정, 기록, 체크포인트"""
        return store.run_detail(q["run"])

    def eval_status(self, q):
        """평가 상태와 체크포인트 목록"""
        s = self.eval.status()
        s["checkpoints"] = store.checkpoints()
        s["rig"] = self.rig.mode
        s["mock"] = self.rig.mock
        return s

    def calib_status(self, q):
        """캘리브레이션 상태와 팔 정보"""
        arm = q.get("arm", "follower")
        return {**self.calib.status(), "info": self.calib.info(arm), "rig": self.rig.mode, "mock": self.rig.mock}

    def estop(self, b):
        """비상 정지"""
        self.rig.estop()

    def connect(self, b):
        """장치 연결"""
        self.rig.connect()

    def disconnect(self, b):
        """평가와 명령 모델 정리 후 장치 연결 해제"""
        self.eval.end()
        self.command.end()
        self.rig.disconnect()

    def teleop(self, b):
        """텔레옵 켜기/끄기"""
        self.rig.set_teleop(bool(b.get("on")))

    def record_start(self, b):
        """에피소드 녹화 시작"""
        self.rig.record_start(b.get("task", ""))

    def record_finish(self, b):
        """녹화 결과 저장(성공/실패) 또는 폐기, 저장 시 에피소드 길이 기록"""
        result = b.get("result")
        if result not in ("success", "fail", None):
            raise Busy("result는 success, fail, null 중 하나")
        idx = self.rig.record_finish(result)
        if idx is not None:
            store.set_label(self.rig.repo, idx, length_s=self.rig.last_saved["length_s"])
        return {"episode": idx}

    def record_dataset(self, b):
        """녹화 대상 데이터셋 변경"""
        return {"dataset": self.rig.set_repo(b["repo"])}

    def label(self, b):
        """에피소드 라벨 수정"""
        fields = {k: b[k] for k in ("result", "position", "memo", "deleted", "keep") if k in b}
        store.set_label(b["repo"], int(b["ep"]), **fields)

    def train_start(self, b):
        """학습 시작"""
        self.trainer.start(b["policy"], b["repo"], int(b["steps"]), int(b["batch"]), float(b["lr"]), int(b["save_freq"]))

    def train_resume(self, b):
        """체크포인트에서 이어서 학습"""
        self.trainer.resume(b["checkpoint"], int(b["steps"]) if b.get("steps") else None)

    def train_delete(self, b):
        """체크포인트 하나 삭제, 학습 중이거나 평가, 명령 카드에 쓰는 체크포인트는 거부"""
        run, step = b["run"], int(b["step"])
        if self.trainer.state == "running" and self.trainer.run == run:
            raise Busy("진행 중인 학습은 지울 수 없습니다")
        for using, why in ((self.eval.in_use(), "평가에 쓰고 있는 체크포인트입니다. 평가를 끝낸 뒤 지우세요"),
                           (self.command.in_use(), "명령 카드에 불러온 체크포인트입니다. 모델을 내린 뒤 지우세요")):
            if not using:
                continue
            ckpt = (store.ROOT / using).resolve()
            if (store.TRAIN_DIR / run).resolve() in ckpt.parents and ckpt.parent.name.isdigit() and int(ckpt.parent.name) == step:
                raise Busy(why)
        target, freed = store.delete_checkpoint(run, step)
        log.info("학습 결과 삭제: %s (%.1fMB)", target.relative_to(store.ROOT), freed / 2**20)
        return {"freed": freed}

    def train_stop(self, b):
        """학습 중단"""
        self.trainer.stop()

    def eval_prepare(self, b):
        """평가 모델 준비, 명령 카드 모델이 떠 있으면 거부"""
        if self.command.in_use():
            raise Busy("명령 카드에서 모델을 쓰고 있습니다. 조작 화면에서 모델을 내린 뒤 평가하세요")
        self.eval.prepare(b["checkpoint"], b.get("label") or b["checkpoint"], b.get("task") or C.TASK,
                          b.get("mode") or None)

    def eval_trial_start(self, b):
        """평가 회차 시작"""
        self.eval.start_trial()

    def eval_trial_finish(self, b):
        """평가 회차 결과 기록"""
        self.eval.finish_trial(bool(b["success"]), b.get("reason", ""))

    def eval_end(self, b):
        """평가 종료"""
        self.eval.end()

    def eval_delete(self, b):
        """평가 기록 하나 삭제, 진행 중인 평가는 거부"""
        if self.eval.session and self.eval.session["id"] == b["id"]:
            raise Busy("진행 중인 평가는 지울 수 없습니다")
        store.delete_eval(b["id"])

    def command_status(self, q):
        """명령 카드 상태, 대화 기록, 체크포인트 목록"""
        return {**self.command.status(), "checkpoints": store.checkpoints(), "mock": self.rig.mock,
                "rig": self.rig.mode, "recording": self.rig.rec is not None, "eval": self.eval.in_use()}

    def command_load(self, b):
        """명령 모델 불러오기, 평가 중이면 거부"""
        if self.eval.in_use():
            raise Busy("평가 화면에서 모델을 쓰고 있습니다. 평가를 끝낸 뒤 불러오세요")
        self.command.load(b["checkpoint"], b.get("label") or b["checkpoint"])

    def command_send(self, b):
        """명령 문장으로 정책 실행"""
        self.command.send(b["text"])

    def command_stop(self, b):
        """명령 중지, home이면 시작 자세로 복귀"""
        self.command.stop(go_home=bool(b.get("home")))

    def command_end(self, b):
        """명령 모델 내리기"""
        self.command.end()

    def calib_start(self, b):
        """캘리브레이션 시작"""
        self.calib.start(b["arm"])

    def calib_middle(self, b):
        """중간 자세 확인"""
        self.calib.middle()

    def calib_finish(self, b):
        """범위 기록 완료와 저장"""
        self.calib.finish()

    def calib_cancel(self, b):
        """캘리브레이션 취소"""
        self.calib.cancel()


def make_handler(app):
    """라우팅 표를 묶은 요청 핸들러 생성"""
    get_routes = {
        "/api/overview": app.overview, "/api/rig": app.rig_status, "/api/telemetry": app.telemetry,
        "/api/datasets": lambda q: store.list_datasets(), "/api/dataset": app.dataset, "/api/signal": app.signal,
        "/api/eval/tasks": lambda q: store.task_suggestions(q.get("checkpoint")),
        "/api/record/tasks": lambda q: store.record_task_suggestions(q["repo"]),
        "/api/train": app.train_status, "/api/train/run": app.train_run, "/api/eval": app.eval_status, "/api/calib": app.calib_status,
        "/api/command": app.command_status,
    }
    post_routes = {
        "/api/estop": app.estop, "/api/rig/connect": app.connect, "/api/rig/disconnect": app.disconnect,
        "/api/teleop": app.teleop, "/api/record/start": app.record_start, "/api/record/finish": app.record_finish,
        "/api/record/dataset": app.record_dataset,
        "/api/label": app.label, "/api/train/start": app.train_start, "/api/train/resume": app.train_resume, "/api/train/delete": app.train_delete,
        "/api/train/stop": app.train_stop,
        "/api/eval/prepare": app.eval_prepare, "/api/eval/trial/start": app.eval_trial_start,
        "/api/eval/trial/finish": app.eval_trial_finish, "/api/eval/end": app.eval_end,
        "/api/eval/delete": app.eval_delete,
        "/api/command/load": app.command_load, "/api/command/send": app.command_send,
        "/api/command/stop": app.command_stop, "/api/command/end": app.command_end,
        "/api/calib/start": app.calib_start, "/api/calib/middle": app.calib_middle,
        "/api/calib/finish": app.calib_finish, "/api/calib/cancel": app.calib_cancel,
    }

    class Handler(BaseHTTPRequestHandler):
        """API, 카메라 스트림, 영상 파일, 정적 파일 처리"""
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            """요청마다 찍히는 기본 접근 로그 끄기"""
            pass

        def _json(self, code, data):
            """JSON 응답"""
            body = json.dumps(data, ensure_ascii=False, allow_nan=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _call(self, fn, arg, path):
            """라우트 실행, 예외를 상태 코드로 변환 (Busy 409, 잘못된 요청 400, 그 외 500)"""
            try:
                r = fn(arg)
                # 반환값 없는 명령도 화면에서 성공으로 판단하도록 빈 객체 응답
                self._json(200, _clean({} if r is None else r))
            except Busy as e:
                log.warning("요청 거부 %s: %s", path, e)
                self._json(409, {"error": str(e)})
            except (KeyError, ValueError, FileNotFoundError) as e:
                log.warning("잘못된 요청 %s: %s", path, e)
                self._json(400, {"error": f"잘못된 요청: {e}"})
            except Exception as e:
                log.error(traceback.format_exc())
                self._json(500, {"error": str(e) or type(e).__name__})

        def do_GET(self):
            """조회 API, 카메라 스트림, 영상, 화면 페이지"""
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in get_routes:
                return self._call(get_routes[u.path], q, u.path)
            if u.path.startswith("/cam/"):
                return self._mjpeg(u.path[5:])
            if u.path == "/video":
                try:
                    return self._file(store.video_path(q["repo"], q["key"], q["chunk"], q["file"]))
                except (KeyError, ValueError):
                    return self._json(400, {"error": "잘못된 요청"})
            if u.path in PAGES:
                return self._file(STATIC / "index.html")
            if u.path.startswith("/static/"):
                p = (STATIC / u.path[8:]).resolve()
                if STATIC in p.parents:
                    return self._file(p)
            self._json(404, {"error": "없는 주소"})

        def do_POST(self):
            """명령 API, 본문은 JSON"""
            u = urlparse(self.path)
            if u.path not in post_routes:
                return self._json(404, {"error": "없는 주소"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "JSON이 아닙니다"})
            log.info("명령 %s %s", u.path, json.dumps(body, ensure_ascii=False) if body else "")
            self._call(post_routes[u.path], body, u.path)

        def _file(self, path):
            """파일 응답, 브라우저 영상 탐색용 Range 요청 처리"""
            if not path.is_file():
                return self._json(404, {"error": "파일이 없습니다"})
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].split(",")[0].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = min(int(b), size - 1) if a and b else end
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".js":
                ctype = "text/javascript"
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def _mjpeg(self, name):
            """카메라 MJPEG 스트림, 장치 연결이 끊기면 종료"""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            last = None
            try:
                while app.rig.mode not in ("off",):
                    frame = app.rig.frames.get(name)
                    if frame is not None and frame is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                        last = frame
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return Handler


def _clean(x):
    """JSON에 넣을 수 없는 NaN/inf를 null로 변환"""
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def serve(mock=False, host="127.0.0.1", port=None):
    """서버 실행, 종료 시 장치와 캘리브레이션 정리"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = App(mock)
    port = port or C.WEB_PORT
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    print(f"metarobo: http://localhost:{port}" + ("  (모의 장치)" if mock else ""), flush=True)
    # 실행 설정 요약
    if not mock:
        log.info("팔로워 %s, 리더 %s, 카메라 %s, %d프레임", C.FOLLOWER_PORT, C.LEADER_PORT,
                 ", ".join(f"{n}={i}" for n, i in C.CAMERAS.items()), C.FPS)
    log.info("녹화 데이터셋 %s, 에피소드 %d초", store.record_repo(mock), C.EPISODE_TIME_S)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.eval.end()
        app.command.end()
        app.rig.disconnect()
        app.calib.cancel()
        if app.trainer.proc and app.trainer.proc.poll() is None:
            print("학습 프로세스는 계속 실행됩니다:", app.trainer.run)
