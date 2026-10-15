"""lerobot-train 하위 프로세스 실행과 로그 파싱"""
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import config as C
import store
from rig import Busy

LOG_FREQ = 100
METRIC = re.compile(r"(\w+(?:/s)?):(\S+)")
log = logging.getLogger("metarobo")
TQDM = re.compile(r"Training:.*?\|\s*(\d+)/(\d+)\s*\[[^<\]]*<([\d:]+)")  # "Training: 6%|..| 12/200 [00:30<07:41, ..]"
# 첫 기록 전 준비 단계, lerobot-train 출력 문구 기준
STAGES = [
    ("Creating dataset", "데이터 불러오는 중"),
    ("Creating policy", "모델 준비 중"),
    ("Creating optimizer", "최적화 설정 중"),
    ("Start offline training", "첫 스텝 준비 중"),
]


def _hms(s):
    """"07:41" 같은 시간 문자열을 초로 변환"""
    sec = 0
    for part in s.split(":"):
        sec = sec * 60 + int(part)
    return sec


class Trainer:
    """학습 프로세스 하나와 진행 상태"""
    def __init__(self):
        self.proc = None
        self.run = None
        self.total = 0
        self.state = "idle"  # idle | running | done | failed | stopped
        self.step = 0
        self.eta_s = None
        self.rows = []
        self.tail = deque(maxlen=30)
        self.params = {}
        self.log_freq = LOG_FREQ
        self.started = None
        self.stage = None
        self.offset = 0  # 이어서 학습한 시작 스텝
        self.base = 0  # 이어 받은 기록 줄 수

    def _exe(self):
        """학습 중복 확인과 lerobot-train 경로"""
        if self.proc and self.proc.poll() is None:
            raise Busy("이미 학습 중입니다")
        from so101 import exe
        try:
            return exe("lerobot-train")
        except SystemExit as e:
            raise Busy(str(e))

    def start(self, policy, repo, steps, batch, lr, save_freq):
        """학습 시작, 학습 제외 에피소드 반영"""
        train = self._exe()

        # CUDA 없으면 CPU 학습
        device = C.DEVICE if (C.DEVICE != "cuda" or gpu_info()["name"]) else "cpu"
        run = f"{policy}_{repo.split('/')[-1]}_{datetime.now():%m%d_%H%M}"
        self.log_freq = min(LOG_FREQ, max(1, steps // 50))
        out = store.TRAIN_DIR / run
        args = [
            f"--dataset.repo_id={repo}",
            f"--policy.device={device}",
            "--policy.push_to_hub=false",
            f"--policy.optimizer_lr={lr}",
            f"--steps={steps}",
            f"--batch_size={batch}",
            f"--save_freq={save_freq}",
            f"--log_freq={self.log_freq}",
            f"--output_dir={out}",
            f"--job_name={run}",
            "--wandb.enable=false",
        ]
        # SmolVLA는 smolvla_base에서 미세조정
        args.insert(1, "--policy.path=lerobot/smolvla_base" if policy == "smolvla" else f"--policy.type={policy}")
        # 삭제 표시와 유지하지 않은 실패 에피소드 제외
        # exclude_episodes는 정식 릴리스(0.6.0, 0.6.1)에 없어 포함 목록(episodes)으로 전달
        excluded = store.excluded_episodes(repo)
        keep = [i for i in range(store.total_episodes(repo)) if i not in set(excluded)]
        if excluded:
            if not keep:
                raise Busy("학습할 에피소드가 없습니다. 데이터 화면에서 삭제 표시나 결과를 확인하세요")
            args.append(f"--dataset.episodes=[{','.join(map(str, keep))}]")

        params = {"policy": policy, "repo": repo, "steps": steps, "batch": batch, "lr": lr,
                  "save_freq": save_freq, "episodes": len(keep), "excluded": len(excluded), "device": device}
        self._launch(train, args, run, out, steps, params)

    def resume(self, checkpoint, steps=None):
        """저장된 체크포인트에서 이어서 학습, steps로 목표 스텝 변경 가능"""
        train = self._exe()
        model = (store.ROOT / checkpoint).resolve()
        out = model.parent.parent.parent
        if (model.name != "pretrained_model" or not model.parent.name.isdigit()
                or out.parent != store.TRAIN_DIR.resolve() or not (model / "train_config.json").exists()):
            raise ValueError("잘못된 체크포인트 경로")
        if not (model.parent / "training_state").is_dir():
            raise Busy("이 체크포인트에는 이어서 학습할 상태(training_state)가 없습니다")
        cfg = store._read(model / "train_config.json", {})
        start_step = int(model.parent.name)
        total = steps or cfg.get("steps") or 0
        if total <= start_step:
            raise Busy(f"목표 스텝은 {start_step:,}스텝보다 커야 합니다")

        self.log_freq = cfg.get("log_freq") or LOG_FREQ
        args = ["--resume=true", f"--config_path={model / 'train_config.json'}",
                f"--steps={total}", f"--output_dir={out}"]
        # 이어 받는 스텝 이후 기록은 첫 새 기록이 나올 때 정리
        rows = [r for r in store.run_log(out.name) if r["step"] <= start_step]
        params = {k: v for k, v in (store.run_params(out) or {}).items() if k != "state"}
        params.update(steps=total, resumed_from=start_step)
        self._launch(train, args, out.name, out, total, params, offset=start_step, rows=rows)

    def _launch(self, train, args, run, out, total, params, offset=0, rows=None):
        """lerobot-train 하위 프로세스 시작과 진행 상태 초기화"""
        self.run, self.total, self.eta_s = run, total, None
        self.offset, self.step, self.rows = offset, offset, list(rows or [])
        self.base = len(self.rows)
        self.params = params
        self.started, self.stage = time.time(), "학습 프로그램 시작 중"
        self.tail.clear()
        self.tail.append("> lerobot-train " + " ".join(args))
        self.proc = subprocess.Popen(
            [train, *args], cwd=store.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        self.state = "running"
        log.info("학습 시작: %s%s", run, f" ({offset}스텝에서 이어서)" if offset else "")
        log.info(self.tail[0])
        threading.Thread(target=self._read, args=(self.proc, out), daemon=True).start()

    def _read(self, proc, out):
        """출력 줄 단위 파싱: 진행 막대의 스텝과 남은 시간, 로그의 손실과 학습률"""
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            m = TQDM.search(line)
            if m:  # 로그가 진행 막대 뒤에 붙어 나오는 경우 있음
                # 이어서 학습하면 진행 막대가 남은 스텝 기준이라 시작 스텝을 더함
                n, of = int(m.group(1)), int(m.group(2))
                self.step = n if of == self.total else self.offset + n
                self.eta_s = _hms(m.group(3))
                line = line[line.index("]", m.end()) + 1:].strip() if "]" in line[m.end():] else ""
                if not line:
                    continue
            self.tail.append(line)
            self.stage = next((ko for en, ko in STAGES if en in line), self.stage)
            if "step:" not in line or "loss:" not in line:
                continue
            metrics = dict(METRIC.findall(line[line.index("step:"):]))
            try:
                # 로그의 step은 "12K"처럼 반올림되어 log_freq로 계산
                step = self.offset + (len(self.rows) - self.base + 1) * self.log_freq
                row = {"step": step, "loss": float(metrics["loss"]),
                       "lr": float(metrics.get("lr", "nan")),
                       "step_s": float(metrics.get("step_s", metrics.get("updt_s", "nan"))),
                       "mem_gb": float(metrics["mem_gb"]) if "mem_gb" in metrics else None}
            except (KeyError, ValueError):
                continue
            self.rows.append(row)
            log.info("학습 %s 스텝 %d/%d, 손실 %.4f", self.run, row["step"], self.total, row["loss"])
            out.mkdir(parents=True, exist_ok=True)
            # 학습 설정은 첫 기록과 함께 저장, 폴더를 미리 만들면 lerobot-train이 거부
            if len(self.rows) == self.base + 1:
                self._save_run(out)
                store.save_run_log(self.run, self.rows)  # 이어서 학습이면 시작 스텝 이후 옛 기록 교체
            else:
                with open(out / "metarobo_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        code = proc.wait()
        if self.state == "running":
            self.state = "done" if code == 0 else "failed"
        if (out / "metarobo_run.json").exists():
            self._save_run(out, state=self.state)
        # 실패 시 마지막 출력 함께 기록
        if self.state == "failed":
            log.error("학습 실패: %s (종료 코드 %d)\n%s", self.run, code, "\n".join(list(self.tail)[-10:]))
        else:
            log.info("학습 종료: %s (%s, 종료 코드 %d)", self.run, self.state, code)

    def _save_run(self, out, **extra):
        """학습 설정과 시작 시각, 종료 상태를 metarobo_run.json에 저장"""
        started = datetime.fromtimestamp(self.started).isoformat(timespec="seconds")
        data = {"started": started, **self.params, **extra}  # 이어서 학습은 처음 시작 시각 유지
        (out / "metarobo_run.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def stop(self):
        """학습 프로세스 강제 종료"""
        if not self.proc or self.proc.poll() is not None:
            raise Busy("진행 중인 학습이 없습니다")
        self.state = "stopped"
        log.info("학습 중단 요청: %s", self.run)
        if sys.platform == "win32":  # 데이터로더 워커까지 함께 종료
            subprocess.call(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self.proc.terminate()

    def status(self):
        """화면용 학습 상태, 서버 재시작 후에는 마지막 학습 기록 사용"""
        rows = self.rows
        if self.run is None:
            runs = sorted(store.TRAIN_DIR.glob("*/metarobo_log.jsonl"), key=lambda p: p.stat().st_mtime)
            if runs:
                rows = store.run_log(runs[-1].parent.name)
        last = rows[-1] if rows else None
        remaining = self.eta_s if self.state == "running" else None
        hint = None
        if self.state == "failed" and any("WinError 1314" in line for line in self.tail):
            hint = ("체크포인트는 저장됐지만 checkpoints/last 바로가기를 만들지 못해 학습이 멈췄습니다. "
                    "Windows 설정 > 시스템 > 개발자용에서 개발자 모드를 켜면 해결됩니다.")
        return {
            "hint": hint,
            "state": self.state, "run": self.run, "step": self.step if self.run else (last or {}).get("step", 0),
            "total": self.total, "params": self.params, "remaining_s": remaining,
            "stage": self.stage, "log_freq": self.log_freq, "elapsed_s": time.time() - self.started if self.state == "running" else None,
            "loss": [[r["step"], r["loss"]] for r in rows],
            "recent": rows[-8:][::-1], "tail": list(self.tail)[-12:], "gpu": gpu_info(),
        }


_gpu = None


def gpu_info():
    """GPU 이름과 메모리, 첫 호출 결과 캐시"""
    global _gpu
    if _gpu is None:
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                _gpu = {"name": p.name, "total_gb": p.total_memory / 2**30}
            else:
                _gpu = {"name": None, "total_gb": None}
        except Exception:
            _gpu = {"name": None, "total_gb": None}
    return _gpu
