import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

def show_result(mapx, mapy, cam_id=0):
    cap = cv2.VideoCapture(cam_id)
    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()
        # Applies the undistorion
        u_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imshow('Undistorted', u_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyWindow('Undistorted')


matriz = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

for fname in images:
    gray = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = gray.copy()
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        # cv2.imshow('img', img)
h, w = gray.shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), 5)
print("<", mtx, ">")
show_result(mapx, mapy)
cv2.destroyAllWindows()
