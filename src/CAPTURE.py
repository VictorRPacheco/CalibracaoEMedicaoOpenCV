import cv2
x=7
cap = cv2.VideoCapture(0)
ret, raw = cap.read()
while True:
    ret, raw = cap.read()
    raw = cv2.flip(raw, 1)
    corners_were_found, corners = cv2.findChessboardCorners(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (8, 6), None)

    nome = "teste.jpg"
    if corners_were_found:
        img = cv2.drawChessboardCorners(raw, (8, 6), corners, corners_were_found)
        cv2.imshow(' sequência de calibração', img)
    key = cv2.waitKey(1)
    if key == ord('c'):
        ret,    raw=cap.read()

        key = cv2.waitKey(2000)
        if key == ord('a'):
            cv2.imwrite(str(x) + nome, raw)
            x = x + 1
    cv2.imshow(' sequência de calibração', raw)
    if key == ord('q'):
        cv2.destroyWindow(' sequência de calibração')
        break
cv2.destroyWindow(' sequência de calibração')
cap.release()