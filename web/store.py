"""LeRobot 데이터셋에 성공/실패 칸이 없어 라벨과 평가 기록은 outputs/metarobo/ 아래 JSON에 별도 보관"""
import json
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path

import config as C

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "metarobo"
TRAIN_DIR = ROOT / "outputs" / "train"
_lock = threading.Lock()


def lerobot_home():
    """LeRobot 데이터셋 저장 위치"""
    from lerobot.utils.constants import HF_LEROBOT_HOME
    return Path(HF_LEROBOT_HOME)


def record_repo(mock):
    """녹화 대상 데이터셋 이름, 화면에서 고른 값이 없으면 설정값, 모의 장치는 _mock 접미사"""
    saved = _read(OUT / "settings.json", {}).get("record_repo_mock" if mock else "record_repo")
    return saved or C.DATASET + ("_mock" if mock else "")


def check_record_repo(repo, mock):
    """새 녹화 데이터셋 이름 검사, 모의 장치는 _mock 접미사를 붙이고 실제 장치는 _mock 데이터셋 거부"""
    repo = repo.strip()
    if not re.fullmatch(r"local/[A-Za-z0-9_-]+", repo):
        raise ValueError("데이터셋 이름은 local/ 뒤에 영문, 숫자, _, - 만 쓸 수 있습니다")
    if mock and not repo.endswith("_mock"):
        repo += "_mock"
    if not mock and repo.endswith("_mock"):
        raise ValueError("모의 장치용 데이터셋에는 실제 장치로 녹화할 수 없습니다")
    return repo


def save_record_repo(repo, mock):
    """고른 녹화 데이터셋을 settings.json에 저장"""
    with _lock:
        data = _read(OUT / "settings.json", {})
        data["record_repo_mock" if mock else "record_repo"] = repo
        _write(OUT / "settings.json", data)


def _read(path, default):
    """JSON 읽기, 없거나 깨졌으면 기본값"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path, data):
    """임시 파일에 쓴 뒤 교체하는 JSON 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _labels_path(repo):
    """데이터셋별 라벨 파일 경로"""
    return OUT / "labels" / (repo.replace("/", "__") + ".json")


def labels(repo):
    """{"에피소드 번호(str)": {"result": "success"|"fail", "keep": bool, "deleted": bool, "position": str, "memo": str, "task": str}}"""
    return _read(_labels_path(repo), {})


def set_label(repo, ep, **fields):
    """에피소드 라벨 일부 필드 갱신"""
    with _lock:
        data = labels(repo)
        data.setdefault(str(ep), {}).update(fields)
        _write(_labels_path(repo), data)


def excluded_episodes(repo):
    """학습에서 뺄 에피소드: 삭제 표시한 것과, 데이터에 유지하지 않은 실패 에피소드"""
    return sorted(int(k) for k, v in labels(repo).items()
                  if v.get("deleted") or (v.get("result") == "fail" and not v.get("keep")))


def list_datasets():
    """로컬 데이터셋 목록 (허브 캐시 제외)"""
    home = lerobot_home()
    out = []
    for info in sorted(home.glob("*/*/meta/info.json")):
        root = info.parent.parent
        if root.parts[-2] == "hub":
            continue
        meta = _read(info, {})
        repo = f"{root.parts[-2]}/{root.parts[-1]}"
        out.append({
            "repo": repo,
            "episodes": meta.get("total_episodes", 0),
            "excluded": len(excluded_episodes(repo)),
            "fps": meta.get("fps"),
        })
    return out


def total_episodes(repo):
    """데이터셋 전체 에피소드 수 (meta/info.json)"""
    return _read(_root(repo) / "meta" / "info.json", {}).get("total_episodes", 0)


def _root(repo):
    """데이터셋 루트 경로, LeRobot 홈 밖이면 거부"""
    root = (lerobot_home() / repo).resolve()
    if lerobot_home().resolve() not in root.parents:
        raise ValueError("잘못된 데이터셋 이름")
    return root


def dataset_detail(repo):
    """에피소드 표와 영상 위치, 녹화 세션이 쓰는 중인 parquet은 읽기 불가"""
    import pyarrow.parquet as pq

    root = _root(repo)
    info = _read(root / "meta" / "info.json", None)
    if info is None:
        raise FileNotFoundError(repo)
    video_keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    rows = []
    unreadable = 0
    # 에피소드 메타 parquet 읽기, 쓰는 중인 파일은 건너뜀
    for f in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        try:
            rows += pq.read_table(f).to_pylist()
        except Exception:
            unreadable += 1
    # 에피소드별 영상 구간과 라벨 합치기
    lab = labels(repo)
    eps = []
    for r in sorted(rows, key=lambda r: r["episode_index"]):
        i = r["episode_index"]
        videos = {}
        for k in video_keys:
            videos[k] = {
                "chunk": r.get(f"videos/{k}/chunk_index"),
                "file": r.get(f"videos/{k}/file_index"),
                "from": r.get(f"videos/{k}/from_timestamp"),
                "to": r.get(f"videos/{k}/to_timestamp"),
            }
        eps.append({
            "index": i,
            "length_s": r["length"] / info["fps"],
            "task": (r.get("tasks") or [""])[0],
            "videos": videos,
            "data": {"chunk": r.get("data/chunk_index"), "file": r.get("data/file_index")},
            **{"result": "", "position": "", "memo": "", "deleted": False, "keep": False, **lab.get(str(i), {})},
        })
    return {"repo": repo, "fps": info["fps"], "video_keys": video_keys, "episodes": eps,
            "total_episodes": info.get("total_episodes", 0), "unreadable": unreadable}


def episode_signal(repo, ep, chunk, file):
    """그리퍼 관측/명령 시계열, 그래프용으로 300점 이하로 축소"""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    root = _root(repo)
    info = _read(root / "meta" / "info.json", {})
    path = root / info["data_path"].format(chunk_index=chunk, file_index=file)
    t = pq.read_table(path, columns=["episode_index", "timestamp", "observation.state", "action"])
    t = t.filter(pc.equal(t["episode_index"], ep))
    names = info["features"]["observation.state"]["names"]
    g = names.index("gripper.pos") if "gripper.pos" in names else len(names) - 1
    ts = t["timestamp"].to_pylist()
    state = [s[g] for s in t["observation.state"].to_pylist()]
    action = [a[g] for a in t["action"].to_pylist()]
    step = max(1, len(ts) // 300)
    return {"t": ts[::step], "state": state[::step], "action": action[::step]}


def video_path(repo, key, chunk, file):
    """에피소드 영상 파일 경로"""
    root = _root(repo)
    info = _read(root / "meta" / "info.json", {})
    p = (root / info["video_path"].format(video_key=key, chunk_index=int(chunk), file_index=int(file))).resolve()
    if root not in p.parents:
        raise ValueError("잘못된 경로")
    return p


def checkpoints():
    """저장된 체크포인트 목록과 해당 스텝 손실"""
    out = []
    for model in sorted(TRAIN_DIR.glob("*/checkpoints/*/pretrained_model")):
        step_dir = model.parent.name
        if not step_dir.isdigit():
            continue  # "last"는 같은 체크포인트의 중복
        run = model.parent.parent.parent
        step = int(step_dir)
        loss = None
        for row in run_log(run.name):
            if row["step"] <= step:
                loss = row["loss"]
        out.append({"path": str(model.relative_to(ROOT)), "run": run.name, "step": step, "loss": loss,
                    "resumable": (model.parent / "training_state").is_dir()})
    return sorted(out, key=lambda c: (c["run"], c["step"]), reverse=True)


def run_log(run):
    """학습별 metarobo_log.jsonl 기록"""
    path = TRAIN_DIR / run / "metarobo_log.jsonl"
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def runs():
    """기록이나 체크포인트가 있는 학습 목록, 최근 순"""
    ckpts = checkpoints()
    out = []
    dirs = [d for d in TRAIN_DIR.glob("*") if d.is_dir()] if TRAIN_DIR.exists() else []
    for d in sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        log = run_log(d.name)
        mine = [c for c in ckpts if c["run"] == d.name]
        if not log and not mine:
            continue
        out.append({"run": d.name, "step": log[-1]["step"] if log else None,
                    "loss": log[-1]["loss"] if log else None, "checkpoints": mine})
    return out


def save_run_log(run, rows):
    """학습 기록 파일을 주어진 줄로 다시 쓰기"""
    path = TRAIN_DIR / run / "metarobo_log.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _run_dir(run):
    """학습 폴더 경로, outputs/train 바로 아래가 아니면 거부"""
    d = (TRAIN_DIR / run).resolve()
    if d.parent != TRAIN_DIR.resolve() or not d.is_dir():
        raise ValueError("잘못된 학습 이름")
    return d


def _finite(row):
    """NaN, 무한대 값을 None으로 바꾼 기록 한 줄"""
    return {k: (v if not isinstance(v, float) or v == v and abs(v) != float("inf") else None) for k, v in row.items()}


def run_params(d):
    """학습 폴더의 설정, metarobo_run.json이 없으면 마지막 체크포인트의 train_config.json 사용"""
    params = _read(d / "metarobo_run.json", None)
    if params is None:
        # 설정 파일 이전 학습은 마지막 체크포인트의 train_config.json 사용
        cfgs = sorted(d.glob("checkpoints/[0-9]*/pretrained_model/train_config.json"))
        cfg = _read(cfgs[-1], None) if cfgs else None
        if cfg:
            eps = cfg["dataset"].get("episodes")
            params = {"policy": cfg["policy"].get("type"), "repo": cfg["dataset"].get("repo_id"),
                      "steps": cfg.get("steps"), "batch": cfg.get("batch_size"),
                      "lr": cfg["policy"].get("optimizer_lr"), "save_freq": cfg.get("save_freq"),
                      "episodes": len(eps) if eps else None, "excluded": None,
                      "device": cfg["policy"].get("device")}
    return params


def _size(p):
    """폴더 안 파일 크기 합 (바이트), 바로가기는 따라가지 않음"""
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) if p.is_dir() else 0


def run_detail(run):
    """학습 하나의 설정, 기록, 체크포인트와 용량"""
    d = _run_dir(run)
    ckpts = [c for c in checkpoints() if c["run"] == d.name]
    for c in ckpts:
        step_dir = ROOT / Path(c["path"]).parent
        c["size"], c["state_size"] = _size(step_dir), _size(step_dir / "training_state")
    return {"run": d.name, "params": run_params(d), "log": [_finite(r) for r in run_log(d.name)],
            "checkpoints": ckpts}


def delete_checkpoint(run, step):
    """체크포인트 하나 삭제 후 지운 경로와 바이트 수 반환"""
    d = _run_dir(run)
    steps = [p for p in (d / "checkpoints").glob("[0-9]*") if p.is_dir() and int(p.name) == int(step)]
    if not steps:
        raise ValueError("체크포인트를 찾을 수 없습니다")
    target = steps[0]
    freed = _size(target)
    shutil.rmtree(target)
    # 지운 체크포인트를 가리키던 last 바로가기 정리
    last = d / "checkpoints" / "last"
    if last.is_symlink() and not last.exists():
        last.unlink()
    return target, freed


def dataset_task_counts(repo):
    """데이터셋 작업 문장별 에피소드 수, 많이 쓴 순"""
    import pyarrow.parquet as pq

    by_ep = {}
    for f in sorted((_root(repo) / "meta" / "episodes").glob("*/*.parquet")):
        try:
            for r in pq.read_table(f, columns=["episode_index", "tasks"]).to_pylist():
                by_ep[r["episode_index"]] = (r["tasks"] or [None])[0]
        except Exception:
            continue  # 쓰는 중인 파일은 건너뜀
    # 녹화 중 저장한 에피소드는 parquet을 아직 못 읽어 라벨의 작업 문장 사용
    for i, lab in labels(repo).items():
        if lab.get("task") and int(i) not in by_ep:
            by_ep[int(i)] = lab["task"]
    counts = {}
    for t in by_ep.values():
        if t:
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def dataset_tasks(repo):
    """데이터셋 에피소드의 작업 문장 목록, 많이 쓴 순"""
    return list(dataset_task_counts(repo))


def record_task_suggestions(repo):
    """녹화 작업 문장 추천: 녹화 대상 데이터셋 문장과 개수, 다른 데이터셋 문장, 기본 문장"""
    out, seen = [], set()

    def add(task, source, count=None):
        if task and task not in seen:
            seen.add(task)
            out.append({"task": task, "source": source, "count": count})

    try:
        for t, n in dataset_task_counts(repo).items():
            add(t, "이 데이터셋", n)
    except ValueError:
        pass  # 잘못된 이름이거나 아직 없는 데이터셋
    for d in list_datasets():
        if d["repo"] != repo:
            for t in dataset_tasks(d["repo"]):
                add(t, "다른 데이터셋")
    add(C.TASK, "기본 문장")
    return out


def checkpoint_repo(checkpoint):
    """체크포인트를 학습한 데이터셋 이름, 모르면 None"""
    model = (ROOT / checkpoint).resolve()
    if TRAIN_DIR.resolve() not in model.parents:
        return None
    return (_read(model / "train_config.json", {}).get("dataset") or {}).get("repo_id")


def task_suggestions(checkpoint):
    """작업 문장 추천: 학습 데이터 문장, 이전 평가 문장(최근 순), 기본 문장, 중복 제거"""
    out, seen = [], set()

    def add(task, source):
        if task and task not in seen:
            seen.add(task)
            out.append({"task": task, "source": source})

    repo = checkpoint_repo(checkpoint) if checkpoint else None
    if repo:
        try:
            for t in dataset_tasks(repo):
                add(t, "학습 데이터")
        except (ValueError, OSError):
            pass  # 데이터셋이 지워졌거나 다른 PC에서 학습한 모델
    for s in reversed(evals()):
        add(s.get("task"), "이전 평가")
    add(C.TASK, "기본 문장")
    return out


def evals():
    """평가 세션 전체 기록"""
    return _read(OUT / "evals.json", [])


def new_eval(checkpoint, label, task):
    """평가 세션 생성, 작업 문장 함께 저장"""
    with _lock:
        data = evals()
        session = {"id": datetime.now().strftime("%Y%m%d_%H%M%S"), "checkpoint": checkpoint, "label": label,
                   "task": task, "grid": list(C.EVAL_GRID), "trials": []}
        data.append(session)
        _write(OUT / "evals.json", data)
        return session


def add_trial(session_id, trial):
    """평가 세션에 회차 결과 추가"""
    with _lock:
        data = evals()
        for s in data:
            if s["id"] == session_id:
                s["trials"].append(trial)
                _write(OUT / "evals.json", data)
                return s
        raise KeyError(session_id)
