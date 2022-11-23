from DroneClass import Tello
import cv2
import numpy as np

def init_drone():
    tello = Tello()
    tello.connect()
    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0
    tello.speed = 0
    print(tello.get_battery())
    tello.streamoff()
    tello.streamon()
    return tello


def telloGetFrame(tello, w=360, h=240):
    myFrame = tello.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img


def findFace(img):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(tello, info, w, pid, pError):
    error = info[0][0] - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    # error = info[0][0] - w // 2
    # speed_forward = pid[0] * error + pid[1] * (error - pError)
    # speed_forward = int(np.clip(speed_forward, -100, 100))

    if info[0][0] != 0:
        tello.yaw_velocity = speed
        # tello.for_back_velocity = speed_forward
    else:
        tello.for_back_velocity = 0
        tello.left_right_velocity = 0
        tello.up_down_velocity = 0
        tello.yaw_velocity = 0
        error = 0

    if tello.send_rc_control:
        tello.send_rc_control(tello.left_right_velocity,
                              tello.for_back_velocity,
                              tello.up_down_velocity,
                              tello.yaw_velocity)

    return error



def main():
    pid = (0.5, 0.5, 0)
    pError = 0
    w, h = 360, 240
    tello = init_drone()
    startCounter = 0

    while True:
        if startCounter == 0:
            tello.takeoff()
            startCounter = 1

        # Step 1
        img = telloGetFrame(tello, w, h)
        # Step 2
        img, info = findFace(img)
        # Step 3
        pError = trackFace(tello, info, w, pid, pError)
        cv2.imshow('Image', img)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            tello.land()
            break


if __name__ == "__main__":
    main()
