"""SO-101 모방학습 실행기. LeRobot CLI를 config.py 설정으로 호출한다.

사용 예:
  python so101.py ports
  python so101.py cams
  python so101.py fps
  python so101.py calibrate            # follower, leader 둘 다
  python so101.py teleop
  python so101.py record -n 10         # 이어서 녹화하려면 --resume
  python so101.py train
  python so101.py eval -n 10
"""
import argparse
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import config as C


def exe(name):
    """lerobot 환경의 실행 파일을 찾는다. VS Code가 다른 파이썬을 쓰면 여기서 걸린다."""
    env = Path(sys.executable).parent
    for d in (env / "Scripts", env):
        for suffix in (".exe", ""):
            p = d / (name + suffix)
            if p.exists():
                return str(p)
    found = shutil.which(name)
    if not found:
        sys.exit(f"'{name}'을 찾을 수 없습니다. lerobot 환경의 파이썬으로 실행했는지 확인하세요.\n현재 파이썬: {sys.executable}")
    return found


def run(name, *args):
    cmd = [exe(name), *args]
    print(">", subprocess.list2cmdline(cmd), flush=True)
    code = subprocess.call(cmd)
    if code:
        sys.exit(code)


def robot_args(with_cameras=True):
    args = ["--robot.type=so101_follower", f"--robot.port={C.FOLLOWER_PORT}", f"--robot.id={C.FOLLOWER_ID}"]
    if with_cameras:
        cams = ", ".join(
            f"{name}: {{type: opencv, index_or_path: {idx}, width: {C.WIDTH}, height: {C.HEIGHT}, fps: {C.FPS}}}"
            for name, idx in C.CAMERAS.items()
        )
        args.append(f"--robot.cameras={{ {cams} }}")
    return args


def teleop_args():
    return ["--teleop.type=so101_leader", f"--teleop.port={C.LEADER_PORT}", f"--teleop.id={C.LEADER_ID}"]


def open_cam(idx):
    import cv2
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, C.FPS)
    return cap


def cmd_cams(a):
    """카메라를 하나씩 띄워 번호를 확인한다. n 다음, p 이전, q 종료."""
    import cv2
    idxs, i = a.indices, 0
    while True:
        idx = idxs[i]
        cap = open_cam(idx)
        key = ord("n")
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"camera {idx}: 프레임을 못 읽음")
                break
            cv2.putText(frame, f"Camera {idx}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, "n: next  p: prev  q: quit", (20, C.HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("camera viewer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("n"), ord("p"), ord("q")):
                break
        cap.release()
        if key == ord("q"):
            break
        i = (i + (-1 if key == ord("p") else 1)) % len(idxs)
    cv2.destroyAllWindows()


def cmd_fps(a):
    """config의 카메라를 동시에 열어 실제 fps를 잰다. 녹화 조건과 같게 하려고 동시에 연다."""
    import cv2
    res = {}

    def measure(name, idx):
        cap = open_cam(idx)
        for _ in range(10):
            cap.read()
        n, t0, frame = 0, time.time(), None
        while time.time() - t0 < 3:
            ok, f = cap.read()
            if ok:
                n, frame = n + 1, f
        cap.release()
        res[name] = n / 3
        if frame is not None:
            cv2.imwrite(f"cam_{name}_{idx}.jpg", frame)

    ths = [threading.Thread(target=measure, args=(n, i)) for n, i in C.CAMERAS.items()]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for name, idx in C.CAMERAS.items():
        fps = res.get(name, 0)
        flag = "OK" if fps >= C.FPS * 0.9 else "낮음: USB 포트를 나누거나 해상도를 낮추세요"
        print(f"{name} (camera {idx}): {fps:.1f} fps  {flag}  -> cam_{name}_{idx}.jpg")


def cmd_calibrate(a):
    if a.which in ("follower", "both"):
        run("lerobot-calibrate", *robot_args(with_cameras=False))
    if a.which in ("leader", "both"):
        run("lerobot-calibrate", *teleop_args())


def cmd_teleop(a):
    cams = robot_args(with_cameras=a.show)
    run("lerobot-teleoperate", *cams, *teleop_args(), f"--display_data={str(a.show).lower()}")


def cmd_record(a):
    run(
        "lerobot-record", *robot_args(), *teleop_args(),
        "--display_data=true",
        f"--dataset.repo_id={C.DATASET}",
        f"--dataset.num_episodes={a.n}",
        f"--dataset.single_task={C.TASK}",
        f"--dataset.episode_time_s={C.EPISODE_TIME_S}",
        f"--dataset.reset_time_s={C.RESET_TIME_S}",
        f"--dataset.fps={C.FPS}",
        "--dataset.push_to_hub=false",
        f"--resume={str(a.resume).lower()}",
    )


def run_name():
    return f"{C.POLICY}_{C.DATASET.split('/')[-1]}"


def cmd_train(a):
    out = Path("outputs/train") / run_name()
    if a.resume:
        run("lerobot-train", f"--config_path={out}/checkpoints/last/pretrained_model/train_config.json", "--resume=true")
        return
    run(
        "lerobot-train",
        f"--dataset.repo_id={C.DATASET}",
        f"--policy.type={C.POLICY}",
        f"--policy.device={C.DEVICE}",
        "--policy.push_to_hub=false",
        f"--steps={a.steps or C.TRAIN_STEPS}",
        f"--batch_size={C.BATCH_SIZE}",
        f"--output_dir={out}",
        f"--job_name={run_name()}",
        "--wandb.enable=false",
    )


def cmd_eval(a):
    ckpt = a.checkpoint or f"outputs/train/{run_name()}/checkpoints/last/pretrained_model"
    if not Path(ckpt).exists():
        sys.exit(f"체크포인트가 없습니다: {ckpt}")
    # 평가 데이터셋 이름은 eval_로 시작해야 하고, 매번 새로 만든다
    stamp = datetime.now().strftime("%m%d_%H%M")
    run(
        "lerobot-record", *robot_args(),
        "--display_data=true",
        f"--dataset.repo_id=local/eval_{run_name()}_{stamp}",
        f"--dataset.num_episodes={a.n}",
        f"--dataset.single_task={C.TASK}",
        f"--dataset.episode_time_s={C.EPISODE_TIME_S}",
        f"--dataset.reset_time_s={C.RESET_TIME_S}",
        "--dataset.push_to_hub=false",
        f"--policy.path={ckpt}",
    )


def main():
    p = argparse.ArgumentParser(description="SO-101 모방학습 실행기")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ports", help="로봇 USB 포트 찾기").set_defaults(f=lambda a: run("lerobot-find-port"))

    s = sub.add_parser("cams", help="카메라 하나씩 보기")
    s.add_argument("indices", nargs="*", type=int, default=[0, 1, 2])
    s.set_defaults(f=cmd_cams)

    sub.add_parser("fps", help="config 카메라 동시 fps 측정").set_defaults(f=cmd_fps)

    s = sub.add_parser("calibrate", help="캘리브레이션")
    s.add_argument("which", nargs="?", choices=["follower", "leader", "both"], default="both")
    s.set_defaults(f=cmd_calibrate)

    s = sub.add_parser("teleop", help="텔레옵 연습")
    s.add_argument("--show", action="store_true", help="카메라 화면도 함께 표시")
    s.set_defaults(f=cmd_teleop)

    s = sub.add_parser("record", help="시연 녹화")
    s.add_argument("-n", type=int, default=10, help="녹화할 에피소드 수")
    s.add_argument("--resume", action="store_true", help="기존 데이터셋에 이어서 녹화")
    s.set_defaults(f=cmd_record)

    s = sub.add_parser("train", help="정책 학습")
    s.add_argument("--steps", type=int)
    s.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 이어서 학습")
    s.set_defaults(f=cmd_train)

    s = sub.add_parser("eval", help="실물 평가 (정책이 로봇을 움직임)")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--checkpoint")
    s.set_defaults(f=cmd_eval)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
