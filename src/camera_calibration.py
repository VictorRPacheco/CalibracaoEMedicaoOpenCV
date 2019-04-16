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

ponto_1_r = None
ponto_2_r = None
ponto_1 = None
ponto_2 = None


def Posicao_e_Distancia(event, x, y, flags, param):
    global clickPoint
    global ponto_1
    global ponto_2
    global raw

    if event == cv2.EVENT_LBUTTONDOWN:
        clickPoint = [x, y]

        if ponto_1 is not None and ponto_2 is not None:
            ponto_1 = None
            ponto_2 = None
        if ponto_1 is None:
            print("set first point")
            ponto_1 = (x, y)
            image = raw.copy()
        elif ponto_2 is None:
            ponto_2 = (x, y)
            distancia = (((ponto_2[0] - ponto_1[0]) ** 2) + ((ponto_2[1] - ponto_1[1]) ** 2)) ** 0.5
            print("---------------------")
            print("Comprimento da reta em pixels: ", distancia)


def Posicao_e_Distancia_raw(event, x, y, flags, param):
    global clickPoint
    global ponto_1_r
    global ponto_2_r
    global raw

    if event == cv2.EVENT_LBUTTONDOWN:
        clickPoint = [x, y]

        if ponto_1_r is not None and ponto_2_r is not None:
            ponto_1_r = None
            ponto_2_r = None
        if ponto_1_r is None:
            print("set first point")
            ponto_1_r = (x, y)
        elif ponto_2_r is None:
            ponto_2_r = (x, y)
            distancia = (((ponto_2_r[0] - ponto_1_r[0]) ** 2) + ((ponto_2_r[1] - ponto_1_r[1]) ** 2)) ** 0.5
            print("---------------------")
            print("Comprimento da reta em pixels: ", distancia)

cv2.namedWindow("Undistorted")
cv2.setMouseCallback("Undistorted", Posicao_e_Distancia)
cv2.namedWindow("Raw")
cv2.setMouseCallback("Raw", Posicao_e_Distancia_raw)


def fill_data(images, objpoints, imgpoints, debug=True, hcenters=H_CENTERS, vcenters=V_CENTERS):
    # This function fills the objpoints and imgpoints list the objpoints elements
    # will be overwritten in the future
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((hcenters * vcenters, 3), np.float32)
    objp[:, :2] = np.mgrid[0:hcenters, 0:vcenters].T.reshape(-1, 2)
    for fname in images:
        gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if debug:
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Find the chessboard centers
        corners_were_found, corners = cv2.findChessboardCorners(gray, (hcenters, vcenters), None)
        if corners_were_found:
            objpoints.append(objp)
            # refines de corner locations
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            if debug:
                cv2.drawChessboardCorners(img, (hcenters, vcenters), corners, corners_were_found)
        if debug:
            cv2.imshow('window', img)
            cv2.waitKey(1000)
    if debug:
        cv2.destroyWindow('window')

def get_extrinsics(image, cam_mat, dist, hcenters=H_CENTERS, vcenters=V_CENTERS):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = np.zeros((hcenters * vcenters, 3), np.float32)
    objpoints[:, :2] = np.mgrid[0:hcenters*27:27, 0:vcenters*27:27].T.reshape(-1, 2)
    print(objpoints)
    gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # Find the chessboard centers
    corners_were_found, corners = cv2.findChessboardCorners(gray, (hcenters, vcenters), None)
    if corners_were_found:
        # refines de corner locations
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret, r, t = cv2.solvePnP(objpoints, corners, cam_mat, dist)

    return r, t


def store_matrices(intrinsics, distortion, n=0):
    mtx_F = cv2.FileStorage("intrisics_"+str(n)+".xml", cv2.FILE_STORAGE_WRITE)
    mtx_F.write("F", intrinsics)
    mtx_F.release()
    mtx_dist = cv2.FileStorage("distortion_"+str(n)+".xml", cv2.FILE_STORAGE_WRITE)
    mtx_dist.write("DIST", distortion)
    mtx_dist.release()


def get_error(objp, imgpoints, rvecs, tvecs, mtx, distortion_vector):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objp[i], rvecs[i], tvecs[i], mtx, distortion_vector)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    return mean_error/len(objpoints)


def show_result(mapx, mapy, cam_id=CAMERA_ID):
    cap = cv2.VideoCapture(cam_id)
    ret, raw = cap.read()

    while ret:
        ret, raw = cap.read()
        raw=cv2.flip(raw, 1)
        # Applies the undistorion
        u_raw = cv2.remap(raw, mapx, mapy, cv2.INTER_LINEAR)
        if ponto_1 is not None and ponto_2 is not None:
            cv2.line(u_raw, ponto_1, ponto_2, (0, 0, 255), 1)
        cv2.imshow('Undistorted', u_raw)

        if ponto_1_r is not None and ponto_2_r is not None:
            cv2.line(raw, ponto_1_r, ponto_2_r, (0, 0, 255), 1)
        cv2.imshow("Raw", raw)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def capture_five_pat(x):
    cap = cv2.VideoCapture(0)
    ret, raw = cap.read()
    seq = x
    while ret and x < (seq+5):
        ret, raw = cap.read()
        cv2.imshow(str(seq/5)+' sequência de calibração', raw)
        key = cv2.waitKey(1)
        if key == ord('c'):
            corners_were_found, corners = cv2.findChessboardCorners(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (H_CENTERS, V_CENTERS), None)
            if corners_were_found:
                nome = ".jpg"
                cv2.imwrite(str(x) + nome, raw)
                x = x + 1
        if key == ord('q'):
            cv2.destroyWindow(str(seq // 5) + ' sequência de calibração')
            break
    cv2.destroyWindow(str(seq//5)+' sequência de calibração')
    cap.release()


def get_num(elem):
    return int(elem.replace(".jpg", ""))


def load_matrix(file, mat_name):
    file_mtx = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
    matrix = file_mtx.getNode(mat_name).mat()
    file_mtx.release()
    return matrix


def get_matrices_mean(q, file_pattern, name):
    mtxs=[]
    for x in range(q):

        matrix = load_matrix(file_pattern%x, name)
        if x == 0:
            mtxs.append(matrix)
        else:
            mtxs[0] += matrix
    mtxs[0] /= q
    return mtxs[0]


def get_matrix_mean(q, matrix):
    for x in range(1, q):
        matrix[0] += matrix[x]
    matrix[0] /= q
    return matrix[0]


if __name__ == "__main__":

    rets = []
    intrinsics = []
    mtxs=[]
    distortion_vector = []
    rvecss = []
    tvecss = []

    cap = cv2.VideoCapture(0)
    ret, raw = cap.read()
    nome = ".jpg"
    cv2.imwrite("tamanho" + nome, raw)
    cap.release()
    h, w = cv2.imread("tamanho.jpg", cv2.IMREAD_GRAYSCALE).shape
    # output image size after the lens correction
    nh, nw = int(SCALE * h), int(SCALE * w)

    intrinsics_matrices_names = glob.glob("intrisics_*.xml")
    distortion_vectors_names = glob.glob("distortion_*.xml")
    if len(intrinsics_matrices_names) > 4 and len(distortion_vectors_names) > 4:
        intrinsics = get_matrices_mean(len(intrinsics_matrices_names), "intrisics_%d.xml", "F")
        distortion_vector = get_matrices_mean(len(distortion_vectors_names), "distortion_%d.xml", "DIST")

    else:
        # Lists to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        for x in range(0, 5):
            capture_five_pat(5*x)
            images = glob.glob('[0-9]*.jpg')
            images.sort(key=get_num)
            images = images[(5*x):(5*(x+1)):1]
            fill_data(images, objpoints, imgpoints, False)
            print("Calculating correction matrices")
            ret, mtx, dist, rvecs, tvecs = \
                cv2.calibrateCamera(objpoints,
                                    imgpoints,
                                    cv2.imread(images[0], cv2.IMREAD_GRAYSCALE).shape[::-1],
                                    None,
                                    None)
            rets.append(ret)
            intrinsics.append(mtx)
            distortion_vector.append(dist)
            rvecss.append(rvecs)
            tvecss.append(tvecs)

            print(mtx)
            # get undistorion maps
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (nw, nh))
            store_matrices(mtx, dist, x)
            print("STD Error:", get_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist))
        intrinsics = get_matrix_mean(5, intrinsics)
        distortion_vector = get_matrix_mean(5, distortion_vector)
    print(mtxs)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(intrinsics, distortion_vector, (w, h), 1, (nw, nh))
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, distortion_vector, None, intrinsics, (nw, nh), 5)

    imgs = glob.glob("0.jpg")

    r, t = get_extrinsics("0.jpg", intrinsics, distortion_vector)

    extrinsics = np.concatenate((r, t), axis=1)

    P = np.dot(intrinsics, extrinsics)
    print("Showing the result")
    # print(newcameramtx)
    show_result(mapx, mapy)

def calculate_extrinsics(mtx, dist):
    for fname in glob.glob('*.jpg'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
