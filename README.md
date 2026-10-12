# SO-101 모방학습

LeRobot CLI를 `config.py` 설정 하나로 실행하는 래퍼입니다.

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
| 8 | `eval -n 10` | 정책이 로봇을 움직이므로 비상정지 준비 |

데이터셋은 `%USERPROFILE%\.cache\huggingface\lerobot\local\` 아래에 저장됩니다.
