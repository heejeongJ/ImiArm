# SO-101 모방학습

LeRobot CLI를 `config.py` 설정 하나로 실행하는 래퍼입니다.

지원 버전: LeRobot 0.6.0 이상 정식 릴리스 (0.6.0, 0.6.1에서 확인). 설치 버전은 `python -c "import importlib.metadata as m; print(m.version('lerobot'))"`로 확인합니다.
학습에는 `lerobot[training]` 추가 패키지와 conda-forge FFmpeg가 필요합니다 (`pip install "lerobot[training]"`, `conda install -c conda-forge ffmpeg`).
0.6.0에서는 `record`가 데이터셋 이름 뒤에 날짜를 붙여 저장합니다. 이때 `train`이 날짜 붙은 이름을 안내하니 `config.py`의 `DATASET`을 그 이름으로 바꾸세요. 0.6.1부터는 이름이 그대로 유지됩니다.

## 준비
1. 이 폴더를 VS Code로 엽니다.
2. `Ctrl+Shift+P` > `Python: Select Interpreter`에서 **lerobot** 환경을 고릅니다.
3. `config.py`의 포트와 카메라 번호를 채웁니다.

## 순서
`Ctrl+Shift+P` > `Tasks: Run Task`에서 번호 순서대로 실행합니다. 터미널에서는 `python so101.py <명령>`으로 실행해도 됩니다.

| 단계 | 명령 | 비고 |
| --- | --- | --- |
| 1 | `ports` | USB를 뽑으라고 할 때 뽑고 Enter |
| 2 | `cams` | n 다음, p 이전, q 종료. 번호를 config.py에 적기 |
| 3 | `fps` | 두 카메라 모두 27fps 이상이어야 함 |
| 4 | `calibrate` | 처음 한 번만 |
| 5 | `teleop` | 동작 전략을 하나로 정하기 |
| 6 | `record -n 10` | 녹화 중 → 다음, ← 다시 녹화, Esc 중단. 이어서 녹화는 `--resume` |
| 7 | `train` | 결과는 `outputs/train/` |
| 8 | `eval -n 10` | 정책이 로봇을 움직이므로 비상정지 준비. `lerobot-rollout`으로 실행되며 키 조작은 녹화와 같음. 결과는 `local/rollout_…` 데이터셋 |

데이터셋은 `%USERPROFILE%\.cache\huggingface\lerobot\local\` 아래에 저장됩니다.

## 웹 화면 (metarobo)
```
python so101.py web           # 실제 장치
python so101.py web --mock    # 로봇 없이 모의 장치로 확인
```
브라우저에서 http://localhost:8000 을 엽니다. VS Code에서는 `0. 웹 화면` 작업으로 실행합니다.

| 화면 | 하는 일 |
| --- | --- |
| 조작 | 장치 연결, 텔레옵, 카메라 영상, 관절 추종 그래프, 시연 녹화 (성공/실패/재녹화) |
| 데이터 | 에피소드 영상 재생, 결과 변경, 물체 위치와 메모, 삭제 표시 |
| 학습 | `lerobot-train` 실행, 손실 곡선, 저장된 체크포인트 |
| 평가 | 체크포인트로 정책 실행, 물체 위치별 성공/실패 기록 |
| 캘리브레이션 | 중간 자세, 가동 범위 기록, 저장을 버튼으로 진행 |
| 시뮬레이션 | 아직 연결 안 됨 |

- 웹 화면은 장치를 한 프로세스에서 직접 엽니다. 웹 화면이 장치에 연결되어 있는 동안에는 위 표의 CLI 명령(`teleop`, `record` 등)을 함께 쓸 수 없습니다.
- 녹화는 `config.py`의 `DATASET` 이름 그대로 이어서 쌓습니다. 모의 장치는 `_mock`을 붙인 별도 데이터셋에 씁니다.
- 성공/실패, 메모, 삭제 표시는 `outputs/metarobo/`에 따로 저장됩니다. 학습할 때 삭제 표시한 에피소드와, 데이터에 유지하지 않은 실패 에피소드는 `--dataset.exclude_episodes`로 뺍니다.
- 비상 정지(또는 Esc)는 정책, 텔레옵, 녹화를 멈추고 지금 자세를 유지합니다. 토크를 끄지는 않습니다.
- Windows에서 학습이 체크포인트 저장 직후 `WinError 1314`로 멈추면 개발자 모드를 켜세요 (LeRobot이 `checkpoints/last` 바로가기를 만듭니다).
