# cam2robot_pos_calibration

```bash
# 0. 환경 (처음 한 번)
cd ~/Projects/xarm_lite6/cam2robot_pos_calibration
source ~/Projects/photoneo-python-examples/GigE-V/harvesters/.venv/bin/activate
pip install -r requirements.txt
echo $GENICAM_GENTL64_PATH

# 1. 카메라 + 체커보드 검출 확인 (data/test_camera_*/detection.png 에서 [0,0] 위치 확인)
python test_camera.py

# 2. 캘리브레이션 데이터 수집 (보드 고정, TCP 끝으로 코너 찍기)
python touch_collect.py

# 3. 계산 -> result/T_base_cam.yaml
python touch_solve.py

# 4. 캘리브레이션 테스트 (보드 코너 위로 이동, t = 오차 측정)
python calib_test.py

# 4-1. 보드 없이 테스트 (작업대 위 X 표시 등을 이미지에서 클릭 -> 그 위로 이동, t = 오차 측정)
python point_test.py

# 5. 카메라 좌표 물체 -> 로봇 이동 테스트 (x y z roll pitch yaw 입력)
python move_obj.py --no-robot
python move_obj.py

# 툴 교체 시 (재캘리브레이션 불필요): config.yaml robot.tcp_offset 수정, 또는 실행 시 지정
python point_test.py robot.tcp_offset="[0,0,0.05,0,0,0]"
python move_obj.py robot.tcp_offset="[0,0,0.05,0,0,0]"

# 6. ROS TF 발행 (result/T_base_cam.yaml 의 ros_static_tf 줄)
grep ros_static_tf result/T_base_cam.yaml
```

```bash
# (보드를 flange 에 달 때) hand-eye 방식: 2~3 대신
python handeye_collect.py --ik-only
python handeye_collect.py
python handeye_solve.py
```
