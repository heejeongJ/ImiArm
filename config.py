# SO-101 실험 설정
# 하드웨어나 실험 조건이 바뀌면 현재 파일만 변경

# 포트: `python so101.py ports`로 확인 (Windows는 COM3 같은 형식)
FOLLOWER_PORT = "COM3"
LEADER_PORT = "COM4"
FOLLOWER_ID = "follower"
LEADER_ID = "leader"

# 카메라: `python so101.py cams`로 번호 확인. 이름은 학습 데이터의 키가 되므로 실험 중 바꾸지 않는다.
CAMERAS = {"front": 1, "wrist": 2}
WIDTH, HEIGHT, FPS = 640, 480, 30

# 데이터셋 (push_to_hub를 끄므로 "local/" 접두사로 로컬에만 저장)
DATASET = "local/so101_pick_pilot"
TASK = "Pick the block and place it in the container"
EPISODE_TIME_S = 30
RESET_TIME_S = 15

# 학습
POLICY = "act"
TRAIN_STEPS = 20000  # 파일럿용. 본 학습은 100000 정도
BATCH_SIZE = 8
DEVICE = "cuda"
