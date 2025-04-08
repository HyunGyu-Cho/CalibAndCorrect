import numpy as np
import cv2

# 저장된 캘리브레이션 파라미터 불러오기
calib_data = np.load("calibration_params.npz")
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

# 왜곡 보정을 테스트할 이미지 파일 경로 (필요에 따라 수정)
image_path = "chessboard_2.png"  # 미리 촬영한 체스보드 이미지 사용
img = cv2.imread(image_path)
if img is None:
    print("오류: 이미지를 불러올 수 없습니다.")
    exit()

h, w = img.shape[:2]
# 최적의 새로운 카메라 행렬 계산 (roi: 관심 영역)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# 이미지 왜곡 보정
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# roi 영역으로 이미지 자르기 (필요 시)
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

# 보정 전후 이미지 비교
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 보정된 이미지 파일로 저장 (원하는 경로와 파일명으로 수정 가능)
cv2.imwrite("undistorted_image.jpg", undistorted_img)
print("보정된 이미지가 undistorted_image.jpg로 저장되었습니다.")
