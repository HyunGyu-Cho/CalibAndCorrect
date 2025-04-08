import numpy as np
import cv2

# 체스보드 내부 코너 수 (열, 행)
chessboard_size = (8, 6)  # 예: 8열 x 6행. 촬영한 체스보드에 맞게 수정하세요.
square_size = 25  # 체스보드 한 칸의 크기 (단위: mm 또는 임의 단위; 일관되게 사용)

# 코너 정밀도를 위한 종료 기준 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 체스보드상의 좌표 (예: (0,0,0), (square_size, 0, 0), …)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# 2차원 격자 좌표 생성 후 square_size만큼 배율 적용
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# 모든 프레임에 대해 객체 좌표와 이미지 좌표를 저장할 리스트 생성
objpoints = []  # 실제 3D 점들
imgpoints = []  # 이미지 상 2D 코너들

# 동영상 파일 열기
video_path = "chessboard.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("오류: 동영상 파일을 열 수 없습니다.")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 너무 많은 프레임을 처리하지 않도록 10 프레임마다 처리
    if frame_count % 10 != 0:
        continue

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret_corners:
        # 코너 정밀도 개선
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)

        # 검출된 코너를 이미지에 표시하여 시각적 피드백 제공
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_corners)
        cv2.imshow('Detected Corners', frame)
        cv2.waitKey(500)  # 500ms 동안 결과 확인

cap.release()
cv2.destroyAllWindows()

if len(objpoints) == 0:
    print("체스보드 코너를 하나도 찾지 못했습니다!")
    exit()

# 카메라 캘리브레이션 수행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print("캘리브레이션 결과:")
print("리프로젝션 에러: ", ret)
print("카메라 행렬:\n", camera_matrix)
print("왜곡 계수: ", dist_coeffs.ravel())

# 전체 리프로젝션 에러(평균) 계산
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("평균 리프로젝션 에러: ", mean_error / len(objpoints))

# 캘리브레이션 결과 파일 저장 (npz 형식)
np.savez("calibration_params.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
print("캘리브레이션 파라미터가 calibration_params.npz 파일에 저장되었습니다.")
