#!/usr/bin/python

# General imports
import numpy as np
import cv2
import json
import glob

# MACROS
CAMERA_ID = 0
H_CENTERS = 8
V_CENTERS = 6
SCALE = 1

def fill_data(images, objpoints, imgpoints, debug=True, hcenters=H_CENTERS, vcenters=V_CENTERS):
    # This function fills the objpoints and imgpoints list the objpoints elements
    # will be overwritten in the future

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((H_CENTERS * V_CENTERS,3), np.float32)
    objp[:,:2] = np.mgrid[0:H_CENTERS, 0:V_CENTERS].T.reshape(-1,2)

    for fname in images:
        frame = cv2.imread(fname)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard centers
        corners_were_found, corners = cv2.findChessboardCorners(gray, (H_CENTERS, V_CENTERS), None)

        if corners_were_found:
            objpoints.append(objp)
            # refines de corner locations
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            if debug == True:
                cv2.drawChessboardCorners(frame, (H_CENTERS, V_CENTERS), corners, corners_were_found)

        if debug == True:
            cv2.imshow('window', frame)
            cv2.waitKey(100)

    if debug == True:
        cv2.destroyWindow('window')

def store_matrices(mat_x, mat_y, file_name):
    params= {}
    params['matrix_x'] = mat_x.tolist()
    params['matrix_y'] = mat_y.tolist()

    file = open(file_name, "w+")
    json.dump(params, file)
    file.close()


def get_error(objp, imgpoints, rvecs, tvecs, mtx, distortion_vector):
    mean_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distortion_vector)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    return mean_error/len(objpoints)

def show_result(mapx, mapy, cam_id=CAMERA_ID):
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

if __name__ == "__main__":

    # Lists to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    x = 0
    while ret:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            nome = ".jpg"
            cv2.imwrite(str(x)+nome,frame)
            x = x+1
        if key == ord('q'):
            break

    images = glob.glob('*.jpg')
    gray = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY)

    fill_data(images, objpoints, imgpoints)

    print "Calculating correction matrices"
    ret, mtx, distortion_vector, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    h, w =  gray.shape
    # output image size after the lens correction
    nh, nw = int(SCALE * h), int(SCALE * w)

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, distortion_vector, (w,h), 1, (nw,nh))
    # get undistorion maps
    mapx,mapy = cv2.initUndistortRectifyMap(mtx, distortion_vector, None, newcameramtx, (nw,nh), 5)

    store_matrices(mapx, mapy, "camera_matrices.json")

    print "STD Error:", get_error(objpoints, imgpoints, rvecs, tvecs, mtx, distortion_vector)

    print "Showing the result"
    show_result(mapx, mapy)