import numpy as np
import cv2
import contourLib
import math
# from computerVision import image_unchanged


def draw(img, corners, imgpts):
    if len(imgpts) > 0 and len(corners) > 0:
        cont = 0
        # if True:
        for corner in corners:
            # corner = corners[0]
            imgpt = np.squeeze(imgpts[cont])
            # imgpt = imgpts[0]
            img_point = np.array((int(imgpt[0]), int(imgpt[1])))
            corner_point = np.array((int(corner[0]), int(corner[1])))
            cv2.line(img, tuple(corner_point), tuple(img_point), (255, 0, 0), 3)
            cont += 1
        # cv2.line(img, (int(corners[0][0]), int(corners[0][1])), (int(imgpts[0][0]), int(imgpts[0][1])), (255, 0, 0))
        return img


def process_image(img):
    # Process image
    image = contourLib.preprocess(img, bilaterial=False, gaussian=False, gaussKernelSize=5, resize=False)
    # Select parts of image that match color of target
    # image = contourLib.hsv(image, (50, 80, 20), (120, 255, 255))
    image = contourLib.hsv(image, (70, 80, 90), (100, 255, 255))
    return image


def rotate_contours(degrees, list_of_points):
    # Convert to radians
    radians = math.radians(degrees)
    # Create affine transform matrix
    rotation_matrix = [[math.cos(radians), -math.sin(radians), 0],
                       [math.sin(radians), math.cos(radians), 0],
                       [0, 0, 1]]
    rotation_matrix = np.array(rotation_matrix)
    rotated_points = []
    for point in list_of_points:
        if type(point) != np.int32:
            # Convert point to list
            if type(point) != list:
                point = np.ndarray.tolist(point)
            # Append 1 to point for matrix multiplication
            if len(point) == 1:
                point = [point[0][0], point[0][1], 1]
            else:
                point = [point[0], point[1], 1]
            # Apply transform
            multiplied = np.matmul(point, rotation_matrix)
            rotated_points.append([multiplied[0], multiplied[1]])
    rotated_points = np.array(rotated_points, dtype=np.float32)

    return rotated_points


def get_outside_corners(contour_inp, is_left):
    cont = np.squeeze(contour_inp)
    # Convert to list
    contour_array = np.ndarray.tolist(cont)

    if not is_left:
        # Locate lowest and innermost points
        farthest_x = [9999999, 0]
        farthest_y = [0, 999999]
        for c in contour_array:
            if c[0] < farthest_x[0]:
                farthest_x = c
            if c[1] < farthest_y[1]:
                farthest_y = c
    else:
        farthest_x = [-999999, 0]
        farthest_y = [0, 999999]
        for c in contour_array:
            if c[0] > farthest_x[0]:
                farthest_x = c
            if c[1] < farthest_y[1]:
                farthest_y = c
    return farthest_x, farthest_y


# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(r):
    rt = np.transpose(r)
    should_be_identity = np.dot(rt, r)
    v = np.identity(3, dtype=r.dtype)
    y = np.linalg.norm(v - should_be_identity)
    return y < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_matrix_to_euler_angles(r):
    assert (is_rotation_matrix(r))

    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])

    singular = sy < 1e-6

    if not singular:
        x_rotation = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x_rotation = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0

    return np.array([x_rotation, y, z])


rotation_vector_list = []
translation_vector_list = []


def get_contour_corners_works(cont):
    # this is from when it worked
    left_rotated = rotate_contours(30, np.squeeze(cont[1]))
    right_rotated = rotate_contours(-30, np.squeeze(cont[0]))
    # Get corners
    left = get_outside_corners([left_rotated], True)
    right = get_outside_corners([right_rotated], False)
    left_rotated = rotate_contours(30, np.squeeze(cont[0]))
    right_rotated = rotate_contours(-30, np.squeeze(cont[1]))
    # Get corners
    left2 = get_outside_corners([left_rotated], True)
    right2 = get_outside_corners([right_rotated], False)
    if left[0][1] > left2[0][1]:
        left = left2
        right = right2
    return left, right


def get_contour_corners_broken(cont):
    left = get_outside_corners(cont[0], True)
    right = get_outside_corners(cont[1], False)
    return left, right


# def get_contour_corners_wip(cont):
#     # left_rotated = rotate_contours(30, np.squeeze(cont[1]))
#     # right_rotated = rotate_contours(-30, np.squeeze(cont[0]))
#     left_rotated = cont[0]
#     right_rotated = cont[1]
#     # Get corners
#     left = get_outside_corners([left_rotated], True)
#     right = get_outside_corners([right_rotated], False)
#     # left_rotated = rotate_contours(30, np.squeeze(cont[0]))
#     # right_rotated = rotate_contours(-30, np.squeeze(cont[1]))
#     # Get corners
#     # right2 = get_outside_corners([left_rotated], True)
#     # left2 = get_outside_corners([right_rotated], False)
#     # if left[0][1] > left2[0][1]:
#     #     left = left2
#     #     right = right2
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
#     # left = np.array(left).reshape(-1, 1, 2)
#     left = np.squeeze(left)
#     grey = cv2.cvtColor(image_unchanged, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("original?", image_unchanged)
#     left = np.ndarray.tolist(left)
#     corners = np.array([left[0], left[1], right[0], right[1]], dtype=np.float32)
#     result = cv2.cornerSubPix(grey,
#                               corners,
#                               (15, 15),
#                               (-1, -1),
#                               criteria)
#     left = [result[0], result[1]]
#     right = [result[2], result[3]]
#     # cv2.imshow("grey", grey)
#     return left, right


def get_contour_corners(cont):
    # ret = get_contour_corners_broken(cont)
    # ret = get_contour_corners_wip(cont)
    ret = get_contour_corners_works(cont)
    return ret




