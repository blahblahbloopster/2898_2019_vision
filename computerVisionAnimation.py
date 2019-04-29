#!/usr/bin/python3

# NOTE: SET THE CAMERA MATRIX CORRECTLY!!

import cv2
import time
import numpy as np
import math
import pickle as pkl
import statistics
from threading import Thread
from computerVisionFunctions import process_image, rotation_vector_list, translation_vector_list, get_contour_corners, \
    rotate_contours

# Options: video, realsense, image, prerecorded
INPUT_DEVICE = "prerecorded"

SERVER_IP = "10.28.98.2"

DISPLAY = True

NETWORK_TABLES = False

FRAME_BY_FRAME = False

DEBUG_PRINT = False

CAMERA_ID = 2

VIDEO_NAME = "example.avi"

mtx = None
dist = None
img_org = None
webcam = None
pipeline = None


class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if NETWORK_TABLES:
    from networktables import NetworkTables
    import logging
    logging.basicConfig(level=logging.DEBUG)
    NetworkTables.initialize(server=SERVER_IP)
    sd = NetworkTables.getTable("SmartDashboard")

if INPUT_DEVICE == "realsense":
    # Semi-deprecated
    import pyrealsense2 as rs
    # Camera pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)


def median_of_vector(vector):
    list_of_item_1 = []
    list_of_item_2 = []
    list_of_item_3 = []
    for b in vector:
        list_of_item_1.append(b[0])
        list_of_item_2.append(b[1])
        list_of_item_3.append(b[2])
    return np.array([statistics.median(list_of_item_1),
                     statistics.median(list_of_item_2),
                     statistics.median(list_of_item_3)])


def solve_pnp_code(contours_inp):
    # pairs = []
    # distances = []
    # for cont in contours_inp:
    if type(contours_inp) != list:
        contours_inp = np.array(contours_inp)
    # filtered_contours = contours_inp
    filtered_contours = contours_inp
    # Rotate, see diagram:
    """
                   / -_
                  /    --__
                 /        /
                /         /
               /         /
              /___       /
                  -_____/   < hard to tell where point is exactly



         /\__
        /    \__
       /        \_
      /           /
     /           /
    /           /
    \__        /
       \____  /
            \/    < more angled, and easier to find bottom corner

    """
    left, right = get_contour_corners([filtered_contours[0], filtered_contours[1]])
    left = np.array(left)
    right = np.array(right)
    # Rotate points back
    left = rotate_contours(-30, left)
    right = rotate_contours(30, right)
    """
    Points of target in px (300 px per inch)
    (413, 0), (996, 149), (0,1599), (584, 1748)
    (1.39, 0), (3.32, 0.497), (0, 5.33), (1.943, 5.827)
    top left, top right, bottom left, bottom right
    (3409, 149), (3992, 0), (3821, 1748), (4405, 1559)
    Points in inches
    (11.363, 0.479), (13.293, 0), (12.74, 5.827), (14.683, 5.33)
    (1.39, 0), (3.32, 0.497), (0, 5.33), (1.943, 5.827), (11.363, 0.479), (13.293, 0), (12.74, 5.827), (14.683, 5.33)
    """
    # PS3 camera distortion matrix
    # matrix2 = [[515.45484128, 0, 285.10931073], [0, 518.05417133, 281.68350735], [0, 0, 1]]
    camera_matrix = np.array(mtx, dtype=np.float32)
    # PS3 camera distortion
    # distortion = np.array([-0.13552493, 0.05373435, 0.0118757, -0.00876742, 0.16312619], dtype=np.float64)
    points = np.array([left, right], dtype=np.float32)
    new_points = []
    points = np.squeeze(points)
    points = np.ndarray.tolist(points)
    # Format points
    for z in points:
        for X in z:
            new_points.append(X)
    new_points = np.array(new_points, dtype=np.float32)
    new_points = np.squeeze(new_points)
    image_points = []
    for d in new_points:
        image_points.append([d])
    image_points = np.array(image_points, dtype=np.int32)

    '''
        objectPoints = [(1.39, 0), (3.32, 0.497), (0, 5.33), (1.943, 5.827), (11.363, 0.479), (13.293, 0),
                             (12.74, 5.827), (14.683, 5.33)]

    '''
    # Real world points in inches
    # object_points = [(3.32, 0.497), (1.943, 5.827), (11.363, 0.479), (12.74, 5.827)]
    # object_points = [(1.39, 0), (0, 5.33), (13.293, 0), (14.683, 5.33)]
    object_points = [(13.249681, 0), (14.626771, 5.324812), (1.37709, 0), (0, 5.324812)]
    # object_points = np.array([[1.39, 0], [3.32, 0.497], [12.74, 5.827], [14.683, 5.33]])
    # objectPoints = np.array([(1.39, 0), (0, 5.53), (13.293, 0), (14.63, 5.33)], dtype=np.float32)
    # Format points
    object_points = np.array(object_points, dtype=np.float32)
    object_points2 = []
    # Move (0, 0) to center
    for pnt in object_points:
        object_points2.append([pnt[0] - (14.683 / 2), pnt[1] - (5.827 / 2)])
    object_points = np.array(object_points2, dtype=np.float32)
    object_points2 = []
    # Add third dimension to points
    for y in object_points:
        object_points2.append([y[0], y[1], 0])
    # objectPoints = np.ascontiguousarray(objectPoints2[:, :3]).reshape((8, 1, 3))
    object_points = np.ascontiguousarray(object_points2)
    # Highlight points being used
    if DISPLAY and img_org:
        cv2.drawContours(img_org, image_points, -1, (255, 255, 200), 10)
        cv2.imshow("highlighted", img_org)
    debug_print("begin solevepnp")
    image_points = np.array(image_points, dtype=np.float32)
    # Do solvepnp
    # print(image_points)
    mystery_value, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist)
    # print("translation")
    # print(translation_vector)
    # print("rotation")
    # print(rotation_vector)
    # Take averages over 5 frames
    translation_vector_list.append(translation_vector)
    rotation_vector_list.append(rotation_vector)
    avg_translation = translation_vector
    avg_rotation = rotation_vector
    if len(translation_vector_list) > 5:
        translation_vector_list.pop(0)
        rotation_vector_list.pop(0)
        avg_rotation = median_of_vector(rotation_vector_list)
        avg_translation = median_of_vector(translation_vector_list)
        # for b in rotation_vector_list:
        #     avg_rotation[0] += b[0] / 5
        #     avg_rotation[1] += b[1] / 5
        #     avg_rotation[2] += b[2] / 5
        # for b in translation_vector_list:
        #     avg_translation[0] += b[0] / 5
        #     avg_translation[1] += b[1] / 5
        #     avg_translation[2] += b[2] / 5
        # avg_rotation = np.array(avg_rotation)
        # avg_translation = np.array(avg_translation)

        # print("average rotation vector")
        # print(np.array(avg_rotation))
        # print("average translation vector")
        # print(np.array(avg_translation))

    return rotation_vector, translation_vector, avg_rotation, avg_translation, image_points


def compute_output_values(rotation_vec, translation_vec):
    # Compute the necessary output distance and angles
    x = translation_vec[0][0] + 0
    z = 0 * translation_vec[1][0] + 1 * translation_vec[2][0]

    # distance in the horizontal plane between robot center and target
    robot_distance = math.sqrt(x**2 + z**2)

    # horizontal angle between robot center line and target
    robot_to_target_angle = math.atan2(x, z)

    rot, _ = cv2.Rodrigues(rotation_vec)
    rot_inv = rot.transpose()

    # version if there is not offset for the camera (VERY slightly faster)
    # pzero_world = numpy.matmul(rot_inv, -tvec)

    # version if camera is offset
    pzero_world = np.matmul(rot_inv, 0 - translation_vec)

    other_angle = math.atan2(pzero_world[0][0], pzero_world[2][0])

    return robot_distance, robot_to_target_angle, other_angle


def debug_print(text):
    if DEBUG_PRINT:
        print(text)
    else:
        return text


if INPUT_DEVICE == "realsense" or INPUT_DEVICE == "video" or INPUT_DEVICE == "prerecorded":
    # Import camera matrix
    with open('camera_calibration2.pkl', 'rb') as f:
        ret, mtx, dist, rotation_vectors, translation_vectors = pkl.load(f)

if INPUT_DEVICE == "image":
    webcam = cv2.imread("my_photo-3.jpg")
    webcam = cv2.resize(webcam, (640, 480))


def cut_image(img, points):
    points2 = points
    if type(points) == list:
        points2 = np.array(points)

    r = cv2.boundingRect(points2)
    return img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]


camera1 = cv2.VideoWriter("camera1_feed.mkv", "X264".encode("utf-8"), fps=30)
camera2 = cv2.VideoWriter("camera2_feed.mkv", "X264".encode("utf-8"), fps=30)
camera3 = cv2.VideoWriter("camera3_feed.mkv", "X264".encode("utf-8"), fps=30)
camera4 = cv2.VideoWriter("camera4_feed.mkv", "X264".encode("utf-8"), fps=30)
camera5 = cv2.VideoWriter("camera5_feed.mkv", "X264".encode("utf-8"), fps=30)
camera6 = cv2.VideoWriter("camera6_feed.mkv", "X264".encode("utf-8"), fps=30)


def vision_code(original_image):
    image_org = None
    if INPUT_DEVICE == "image":
        image_org = webcam

    if INPUT_DEVICE == "video" or INPUT_DEVICE == "prerecorded":
        image_unchanged = original_image
        # if not got_image:
        #     return None, None, None
        image_unchanged = cv2.resize(image_unchanged, (640, 480))
        camera1.write(image_unchanged)
        # print(image_unchanged.size())
        image_unchanged = cv2.undistort(image_unchanged, mtx, dist)
        camera2.write(image_unchanged)
        # image_unchanged.resize((640, 480))
        image_org = image_unchanged
        # image_unchanged = cv2.resize(image_unchanged, (640, 480))
        # image_unchanged = cv2.undistort(image_unchanged, mtx, dist)
    if INPUT_DEVICE == "realsense":
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        image_org = color_image
        cv2.imshow("realsense", image_org)
    debug_print("Begin processing")
    frame = process_image(image_org)
    camera3.write(frame)
    debug_print("Finish processing")
    # Display output of pipeline
    if DISPLAY:
        cv2.imshow("Frame", frame)
    # Get contours
    _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = contours
    debug_print("Got contours")

    # Filter contours based on perimeter-to-area ratio
    simplified = []
    if len(new_contours) > 1:
        matched = []
        good_ratio = []
        for cnt in new_contours:
            if cv2.contourArea(cnt) > 50:
                ratio = cv2.arcLength(cnt, True) / cv2.contourArea(cnt)
                maximum = 0
                minimum = 100000
                for w in cnt:
                    if w[0][0] > maximum:
                        maximum = w[0][0]
                    if w[0][0] < minimum:
                        minimum = w[0][0]
                if ratio < 0.5:
                    # 0.3
                    #  and maximum < 610 and minimum > 80
                    simplified.append(cnt)
                    good_ratio.append(True)
        if len(simplified) > 1:
            closest_contours = []
            contour_list = []
            confirmed_pairs = []
            main_count = 0
            for contour in simplified:
                moments = cv2.moments(contour)
                x_cent = int(moments['m10'] / moments['m00'])
                y_cent = int(moments['m01'] / moments['m00'])
                contour_list.append([x_cent, y_cent, contour])
            # contour_list = sorted(contour_list, key=itemgetter(0))
            for data in contour_list:
                x_cent = data[0]
                closest_value = 9999999
                closest_data = []
                closest_count = 0
                count = 0
                for test in contour_list:
                    x_difference = abs(test[0] - x_cent)
                    if main_count not in matched:
                        if x_difference < closest_value and x_difference != 0:
                            closest_count = count
                            closest_value = x_difference
                            closest_data = test
                    count += 1
                matched.append(main_count)
                matched.append(count)
                confirmed_pairs.append([main_count, closest_count])
                closest_contours.append(closest_data)
                main_count += 1
            double_confirmed = []
            double_confirmed_list = []
            for x in confirmed_pairs:
                if x[0] == confirmed_pairs[x[1]][1]:
                    x_list = contour_list[x[0]][2]
                    n_list = contour_list[x[1]][2]
                    cv2.drawContours(img_org, new_contours, 1, (255, 0, 0), 25)
                    if [x[1], x[0]] not in double_confirmed_list and len(x_list) > 2 and len(n_list) > 2:
                        double_confirmed.append([n_list, x_list])
                        double_confirmed_list.append(x)
            confirmed_contours = double_confirmed
            debug_print("End contour filter")
            if True:
                robot_distance = None
                first_angle = None
                second_angle = None
                for i in confirmed_contours:
                    debug_print("Begin solvepnp")
                    rotation_vector, translation_vector, average_rotation, average_translation, img_pts = \
                        solve_pnp_code(i)
                    debug_print("End solvepnp")
                    # PS3 camera matrix
                    # matrix2 = [[515.45484128, 0, 285.10931073], [0, 518.05417133, 281.68350735], [0, 0, 1]]
                    if DISPLAY:
                        # mult = 10
                        # add = 10
                        # image_org = draw(image_org, [(14.626771 * mult + add, 5.324812 * mult), (13.249681 *
                        # mult + add
                        # , 0),
                        #                          (0 + add, 5.324812 * mult)]
                        #                , img_pts)
                        debug_print("Drawing axis")
                        cv2.aruco.drawAxis(image_org, mtx, dist, rotation_vector, translation_vector, 10)
                        cv2.imshow("axis", image_org)
                    debug_print("Doing Rodriques")
                    # destination = cv2.Rodrigues(rotation_vector)[0]
                    robot_distance, first_angle, second_angle = compute_output_values(average_rotation,
                                                                                      average_translation)
                    debug_print("finished")
                    # print("robot_distance")
                    # print(robot_distance)
                    print(str(math.degrees(first_angle)) + ", " + str(math.degrees(second_angle)))

                    if NETWORK_TABLES:
                        sd.putNumber("robot_distance", robot_distance)
                        sd.putNumber("angle_a", math.degrees(first_angle))
                        sd.putNumber("angle_b", math.degrees(second_angle))
                        NetworkTables.flush()
                    # break
                return robot_distance, first_angle, second_angle
    return None, None, None


repetitions = 0
startTime = time.monotonic()
# for o in list_thing:
if True:
    # distance_list = []
    # angle1_list = []
    # angle2_list = []
    # if not o:
    #     continue
    if INPUT_DEVICE == "prerecorded":
        webcam = cv2.VideoCapture(VIDEO_NAME)

    if INPUT_DEVICE == "video":
        import os
        # video_name = "folder_of_videos/my_video-" + o[0] + "_newer.mkv"
        # webcam = cv2.VideoCapture(video_name)
        # webcam = cv2.VideoCapture(CAMERA_ID)

        os.system("v4l2-ctl --device " + str(CAMERA_ID) + "  -c exposure_auto=1")
        time.sleep(1)
        print("auto")
        os.system("v4l2-ctl --device " + str(CAMERA_ID) + " -c white_balance_temperature_auto=0")
        time.sleep(1)
        print("balance")
        os.system("v4l2-ctl --device " + str(CAMERA_ID) + " -c white_balance_temperature=3004")
        time.sleep(1)
        print("temp")
        # webcam = cv2.VideoCapture(CAMERA_ID)
        time.sleep(1)
        os.system("v4l2-ctl --device " + str(CAMERA_ID) + " -c exposure_absolute=8")
        # webcam.set(15, 8)
        print("exposure")
        time.sleep(1)

        webcam = WebcamVideoStream(CAMERA_ID).start()
        distance = "blah blah blah"
    counter = 0
    while True:
        if FRAME_BY_FRAME:
            while True:
                if cv2.waitKey(1) & 0xFF == ord("x"):
                    break

        if INPUT_DEVICE == "video" or INPUT_DEVICE == "prerecorded":
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        ret, image = webcam.read()
        if counter > 0:
            counter -= 0.1
        if not ret:
            counter += 1
            print("This shouldn't happen (image not captured successfully)")
            if counter > 30:
                print("Exiting code, camera/video not responding")
                exit(1)
            continue
        distance, angle1, angle2 = vision_code(image)

        # if not distance:
        #     continue
        # distance_list.append(distance)
        # angle1_list.append(angle1)
        # angle2_list.append(angle2)
        # time.sleep(0.02)
        repetitions += 1
        if repetitions >= 1000:
            print("average time per cycle = ", (time.monotonic() - startTime) / 1000, "seconds")
            print("average FPS = ", 1 / ((time.monotonic() - startTime) / 1000))
            startTime = time.monotonic()
            repetitions = 0
        if INPUT_DEVICE == "image":
            while True:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if INPUT_DEVICE == "image":
            break
    # if len(distance_list) > 0:
    #     print("dist")
    #     print(float(o[1]) - statistics.median(distance_list))
    #     print("angle")
    #     print(float(o[2]) - statistics.median(angle1_list))

webcam.stop()
cv2.destroyAllWindows()
