#!/usr/bin/python

# General imports
import numpy as np
import cv2
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
    objp = np.zeros((H_CENTERS * V_CENTERS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:H_CENTERS, 0:V_CENTERS].T.reshape(-1, 2)

    for fname in images:
        gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if debug:
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Find the chessboard centers
        corners_were_found, corners = cv2.findChessboardCorners(gray, (H_CENTERS, V_CENTERS), None)

        if corners_were_found:
            objpoints.append(objp)
            # refines de corner locations
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            if debug:
                cv2.drawChessboardCorners(img, (H_CENTERS, V_CENTERS), corners, corners_were_found)

        if debug:
            cv2.imshow('window', img)
            cv2.waitKey(100)

    if debug:
        cv2.destroyWindow('window')


def store_matrices(intrisics, distortion, n=0):
    mtx_F = cv2.FileStorage("intrisics_"+str(n)+".xml", cv2.FILE_STORAGE_WRITE)
    mtx_F.write("F", intrisics)
    mtx_F.release()
    mtx_dist = cv2.FileStorage("distortion_"+str(n)+".xml", cv2.FILE_STORAGE_WRITE)
    mtx_dist.write("DIST", distortion)
    mtx_dist.release()


def get_error(objp, imgpoints, rvecs, tvecs, mtx, distortion_vector):
    mean_error = 0
    for i in range(len(objpoints)):
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


def capture_five_pat(x):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    seq = x
    while ret and x < (seq+5):
        ret, frame = cap.read()
        cv2.imshow(str(seq/5)+' sequência de calibração', frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            corners_were_found, corners = cv2.findChessboardCorners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (H_CENTERS, V_CENTERS), None)
            if corners_were_found:
                nome = ".jpg"
                cv2.imwrite(str(x) + nome, frame)
                x = x + 1
        if key == ord('q'):
            break
    cv2.destroyWindow(str(seq/5)+' sequência de calibração')
    cap.release()

def get_num(elem):
    return int(elem.replace(".jpg", ""))


if __name__ == "__main__":

    rets = []
    mtxs = []
    distortion_vectors = []
    rvecss = []
    tvecss = []

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    nome = ".jpg"
    cv2.imwrite("tamanho" + nome, frame)
    cap.release()
    h, w = cv2.imread("tamanho.jpg", cv2.IMREAD_GRAYSCALE).shape
    # output image size after the lens correction
    nh, nw = int(SCALE * h), int(SCALE * w)

    intrisics_matrices_names = glob.glob("intrisics_*.xml")
    distortion_vectors_names = glob.glob("distortion_*.xml")
    if len(intrisics_matrices_names) > 4 and len(distortion_vectors_names) > 4:
        print("have parameters")
        for x in range(len(intrisics_matrices_names)):
            file_mtx = cv2.FileStorage("intrisics_"+str(x)+".xml", cv2.FILE_STORAGE_READ)
            matrix = file_mtx.getNode("F").mat()
            file_mtx.release()
            if(len(mtxs)==0):
                mtxs.append(matrix)
            else:
                mtxs[0] += matrix
        mtxs[0] /= len(intrisics_matrices_names)
        for x in range(len(distortion_vectors_names)):
            file_mtx = cv2.FileStorage("distortion_"+str(x)+".xml", cv2.FILE_STORAGE_READ)
            matrix = file_mtx.getNode("DIST").mat()
            file_mtx.release()
            if (len(distortion_vectors) == 0):
                distortion_vectors.append(matrix)
            else:
                distortion_vectors[0] += matrix
        distortion_vectors[0] /= len(distortion_vectors_names)

    else:
        # Lists to store object points and image points from all the images.
        for x in range(0, 5):
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.
            capture_five_pat(5*x)
            images = glob.glob('[0-9]*.jpg')
            images.sort(key=get_num)
            images = images[(5*x):(5*(x+1)):1]
            fill_data(images, objpoints, imgpoints, False)

            print("Calculating correction matrices")
            ret, mtx, distortion_vector, rvecs, tvecs = \
                cv2.calibrateCamera(objpoints,
                                    imgpoints,
                                    cv2.imread(images[0], cv2.IMREAD_GRAYSCALE).shape[::-1],
                                    None,
                                    None)

            rets.append(ret)
            mtxs.append(mtx)
            distortion_vectors.append(distortion_vector)
            rvecss.append(rvecs)
            tvecss.append(tvecs)

            print(mtx)
            # get undistorion maps
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distortion_vector, (w, h), 1, (nw, nh))
            store_matrices(newcameramtx, distortion_vector, x)
            print("STD Error:", get_error(objpoints, imgpoints, rvecs, tvecs, mtx, distortion_vector))
    print(mtxs[0])
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtxs[0], distortion_vectors[0], (w,h), 1, (nw, nh))
    mapx, mapy = cv2.initUndistortRectifyMap(mtxs[0], distortion_vectors[0], None, newcameramtx, (nw, nh), 5)
    print("Showing the result")
    print(newcameramtx)
    show_result(mapx, mapy)
