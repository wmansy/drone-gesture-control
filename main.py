import threading
import traceback

import cv2
import mediapipe as mp
import numpy as np
from piosdk import Pioneer
from collections import namedtuple
import time


def eql(num1, num2, err=10):
    if num1 < 0 or num2 < 0:
        return True
    return True if abs(num1 - num2) <= err else False


def eql_all(left=None, right=None):  # , neck=[]):
    ans = True
    if left:
        ans = (
                eql(body.left_collarbone.angle, left[0]) and
                eql(body.left_arm.angle, left[1]) and
                eql(body.left_forearm.angle, left[2]) and
                ans)
    if right:
        ans = (
                eql(body.right_collarbone.angle, right[0]) and
                eql(body.right_arm.angle, right[1]) and
                eql(body.right_forearm.angle, right[2]) and
                ans)
    # if neck:
    #     ans = (eq(parts.neck.angle) and
    #            ans)
    return ans


def remap(oldValue, oldMin, oldMax, newMin, newMax):
    """
    Функция для преобразования числовых значений из одного диапазона в другой
    """
    oldRange = (oldMax - oldMin)
    if (oldRange == 0):
        newValue = newMin
    else:
        newRange = (newMax - newMin)
        newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin
    return newValue


def ang(v1):
    """
    Функция рассчитывает направление вектора на плоскости и возвращает угол от 0 до 359
    """
    angle = round(np.degrees(np.arctan2(v1[1], -v1[0])))
    angle = remap(angle, -180, 179, 0, 359)
    return round(angle)


def convert_points(points):
    """
    Функция для конвертации определенных нейросетью точек скелета
    из относительных координат (0-1.0) в абсолютные координаты
    """
    converted_points = []

    base_point = Point(x=round(IMGW * (points[12].x + points[11].x) // 2),
                       y=round(IMGH * (points[12].y + points[11].y) // 2),
                       z=(points[12].z + points[11].z) / 2,
                       visibility=(points[12].visibility + points[11].visibility) / 2)
    for p in points:
        converted_points.append(Point(x=round(IMGW * p.x),
                                      y=round(IMGH * p.y),
                                      z=p.z,
                                      visibility=p.visibility))
    converted_points.append(base_point)
    return converted_points


def generate_parts_vectors(pts):
    """
    Функция для представления частей тела в виде векторов.
    Принимает набор точек, а возвращает вектора.
    """
    j = {}
    for joint in body_parts.items():
        pos = joint[1]
        vec_x = pts[pos[1]].x - pts[pos[0]].x
        vec_y = pts[pos[1]].y - pts[pos[0]].y
        j.update({
            joint[0]: Part(vec_x, vec_y, ang([vec_x, vec_y]))
        })
    j = Parts(**j)
    return j


def detect():
    global cordX, cordY, pose_detected
    if eql_all(left=[180, 270, 45], right=[0, 270, 135]):
        print("\n\n\nPOSE 1\n\n\n")
        # if take_photo_time == -1:
        #     print("Picture will be saved in 5 seconds")
        #     take_photo_time = time.time()
        pose_detected = time.time()

    elif eql_all(left=[180, 180, 180]):
        print("\n\n\nPOSE 2\n\n\n")
        cordX += stepXY
        pose_detected = time.time()

    elif eql_all(right=[0, 0, 0]):
        print("\n\n\nPOSE 3\n\n\n")
        cordX -= stepXY
        pose_detected = time.time()

    elif eql_all(left=[180, 180, 90]):
        print("\n\n\nPOSE 4\n\n\n")
        cordY += stepXY
        pose_detected = time.time()

    elif eql_all(right=[0, 0, 90]):
        print("\n\n\nPOSE 5\n\n\n")
        cordY -= stepXY
        pose_detected = time.time()

    elif eql_all(left=[180, 225, 270], right=[0, 315, 270]):
        print("\n\n\nPOSE 6\n\n\n")
        pose_detected = time.time()
        if not useIntegratedCam:
            pioneer.land()


def bpla(converted_points):
    global cordX, cordY, cordZ, pose_detected
    global yaw_err, yaw_errold, yaw_kp, yaw_kd, yaw_k
    global z_err, z_errold, z_kp, z_kd
    global y_err, y_errold, y_kp, y_kd
    global yaw
    global cordX, cordY, cordZ, yaw, pose_detected
    global yaw_errold, z_errold, y_errold

    yaw_err = -(IMGW // 2 - converted_points[33].x) * yaw_k
    yaw_u = yaw_kp * yaw_err - yaw_kd * (yaw_err - yaw_errold)
    yaw_errold = yaw_err

    y_err = -(-0.15 - converted_points[33].z)
    y_u = y_kp * y_err - y_kd * (y_err - y_errold)
    y_errold = y_err

    z_err = (IMGH // 2 - converted_points[33].y)
    z_u = z_kp * z_err - z_kd * (z_err - z_errold)
    z_errold = z_err

    yaw += yaw_u
    cordY += y_u
    cordZ += z_u
    pioneer.go_to_local_point(cordX, cordY, cordZ, yaw=yaw)


def main():
    global cap, pioneer
    global pose_detected, body, Point, IMGW, IMGH, cordY, cordZ, cordX

    take_photo_time = -1
    pose_detected = -1

    mpDrawings = mp.solutions.drawing_utils
    skeletonDetectorConfigurator = mp.solutions.pose
    skDetector = skeletonDetectorConfigurator.Pose(static_image_mode=False,
                                                   min_tracking_confidence=0.8,
                                                   min_detection_confidence=0.8,
                                                   model_complexity=2)

    converted_points = []
    gg = time.time()

    try:
        if useIntegratedCam:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            pioneer = Pioneer()

        while True:
            if useIntegratedCam:
                ret, frame = cap.read()
            else:
                img = pioneer.get_raw_video_frame()
                frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
                # print("\n\n\n\n\n\n\n\n\nfgffgfggfgfg")
            frame = cv2.flip(frame, 1)

            IMGW = np.shape(frame)[1]
            IMGH = np.shape(frame)[0]

            detected_skeletons = skDetector.process(frame)

            if detected_skeletons.pose_landmarks is not None:

                converted_points = convert_points(detected_skeletons.pose_landmarks.landmark)
                body = generate_parts_vectors(converted_points)

                cv2.circle(frame, (converted_points[33].x, converted_points[33].y), 4, (255, 0, 0), 3)

                if pose_detected == -1:
                    threading.Thread(target=detect).start()

                elif pose_detected != -1 and time.time() - pose_detected > 1:
                    pose_detected = -1

                # if take_photo_time != -1 and time.time() - take_photo_time > 5:
                #     cv2.imwrite("image", frame)
                #     take_photo_time = -1
                #     pose_detected = -1

            if not useIntegratedCam and converted_points:
                threading.Thread(target=bpla, args=(converted_points,)).start()
                converted_points.clear()

            mpDrawings.draw_landmarks(frame, detected_skeletons.pose_landmarks,
                                      skeletonDetectorConfigurator.POSE_CONNECTIONS)

            cv2.imshow("frame", frame)

            key = cv2.waitKey(1) & 0xff

            if key == ord('q') or key == 27:
                if not useIntegratedCam:
                    pioneer.command_id = 0
                    pioneer.disarm()
                break

            elif key == ord('a'):
                pioneer.command_id = 0
                pioneer.arm()

            elif key == ord('l') and not useIntegratedCam:
                pioneer.command_id = 0
                pioneer.land()
                cordX, cordY = 0, 0
                cordZ = 1.5

            elif key == ord('j') and not useIntegratedCam:
                pioneer.command_id = 0
                pioneer.takeoff()
            elif key == ord('p'):
                cv2.imwrite("image", frame)
            # elif key == ord('u'):
            #     cordZ += stepXY

    except Exception as e:
        print('Ошибка:\n', traceback.format_exc())
        pioneer.land()
        time.sleep(3)
        pioneer.disarm()

    if useIntegratedCam:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    useIntegratedCam = False
    cordX = .0
    cordY = .0
    cordZ = 1.5
    yaw = np.radians(0)

    stepXY = 0.5

    yaw_err = 0
    yaw_errold = 0
    yaw_kp = .005
    yaw_kd = .0025
    yaw_k = 0.01

    z_err = 0
    z_errold = 0
    z_kp = .00004
    z_kd = .00001

    y_err = 0
    y_errold = 0
    y_kp = .12
    y_kd = .01

    body_parts = {"neck": [33, 0],
                  "left_collarbone": [33, 12],
                  "left_arm": [12, 14],
                  "left_forearm": [14, 16],
                  "right_collarbone": [33, 11],
                  "right_arm": [11, 13],
                  "right_forearm": [13, 15]}

    Point = namedtuple('Point', 'x y z visibility')

    Parts = namedtuple("Parts", body_parts.keys())

    Part = namedtuple("Part", 'x y angle')
    main()
