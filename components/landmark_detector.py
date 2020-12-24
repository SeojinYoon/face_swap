# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:38:16 2020

@author: seojin
"""

"""
component about landmark detection
"""

import dlib
import numpy as np
import cv2
from components.sj_image_process import sobel_edge_img
from components.sj_keras_toolbox import find_faceRoi_withParser
from components.sj_util import partition_d1_2
import copy
from components.sj_geometry import SJ_ellipse
from components.sj_keras_toolbox import find_faceRoi, face_parsing_index


class DlibFaceNotFoundException(Exception):
    """
    exception that Dlib is not detect face
    """
    pass


class DlibLandmarkNotFoundException(Exception):
    """
    exception that Dlib is not detect face landmark
    """
    pass


def dlib_landmark_faceArea(landmark_pts, face_area):
    """
    extract landmarks about specific face area

    :param landmark_pts: landmark points (dlib point list)
    :param face_area: face area(chin, left eyebrow, right eyebrow, upper nose, side of nose, left eye, right eye, lips outline, lips inline, left eye left_end, left eye right_end, right eye left_end, right eye right_end, face center)
    :return: landmarks
    """
    parser = dlib_detection_parser()[face_area]
    lower_index = parser[0]
    upper_index = parser[1]

    return [landmark_pts[i] for i in range(lower_index, upper_index + 1)]


def dlib_point2tuple(pt):
    """
    map point to tuple

    :param pt: dlib's point
    :return: position(tuple)
    """
    return (pt.x, pt.y)


def dlib_points2tuple(pts):
    """
    map points to tuple
    :param pts: dlib's points
    :return: list of position(tuple)
    """
    return list(map(lambda pt: (pt.x, pt.y), pts))


def landmark_scaling(landmarks, x_scale, y_scale):
    """
    scaling landmark

    :param landmarks: dlib points
    :param x_scale: scale factor about x-axis
    :param y_scale: scale factor about y-axis
    :return: dlib points scaled by scale factor
    """
    return [dlib.point(x=int(l_m.x * x_scale), y=int(l_m.y * y_scale)) for l_m in landmarks]


def dlib_detection_parser():
    """
    landmark's range of index about face area

    :return: landmark's range of index about face area
    """
    return {
        'chin': [0, 16],
        'left eyebrow': [17, 21],
        'right eyebrow': [22, 26],
        'upper nose': [27, 30],
        'side of nose': [31, 35],
        'left eye': [36, 41],
        'right eye': [42, 47],
        'lips outline': [48, 59],
        'lips inline': [60, 66],
        'left eye left_end': 36,
        'left eye right_end': 39,
        'right eye left_end': 42,
        'right eye right_end': 45,
        'face center': 33,
    }


def dlib_indexes(face_area):
    """
    extract dlib indexes correspond to face area

    :param face_area: interst face area(chin, left eyebrow, right eyebrow, upper nose, side of nose, left eye, right eye, lips outline, lips inline, left eye left_end, left eye right_end, right eye left_end, right eye right_end, face center)
    :return: dlib index list correspond to face area
    """
    parser = dlib_detection_parser()[face_area]
    lower_index = parser[0]
    upper_index = parser[1]

    return [i for i in range(lower_index, upper_index + 1)]


def dlib_landmark_range():
    """
    dlib landmark's index range

    :return: dlib landmark point's index range
    """
    return [0, 67]


def redefine_face_border_landmarks(img, parser, landmarks):
    """
    redefine face's chin line

    How to redefine chin line?
    -> 1. detect landmarks about face using dlib
    -> 2. extract face area using face parser from image
    -> 3. apply sobel filter to find border of face
    -> 4. global fixed binarization is applied to extract clear images.
    -> 5. reposition dlib's landmarks about chin landmark, mapping to face border

    :param img: face image
    :param parser: face parser
    :param landmarks: landmarks(dlib point list)
    :return: image which redefined chin line of face
    """
    variable_landmarks = copy.deepcopy(landmarks)

    img_height, img_width = img.shape[0], img.shape[1]

    # face_roi -> sobel edge -> thresholding -> landmark positioning
    roi_img = find_faceRoi_withParser(parser, img)

    edge_img = sobel_edge_img(roi_img)
    edge_img = cv2.threshold(edge_img, 200, 255, cv2.THRESH_BINARY)[1]

    # 엣지 이미지의 픽셀과 일치하는 정확한 landmark의 포지션을 잡아야함(landmark가 가장 많이 빗나가는 0~4, 13~16 에 대하여 수행한다.)
    # 255인 값의 픽셀중에서 landmark의 y포지션과 동일한 곳으로 이동한다. 이때 너무 이동되는것을 방지하기 위해 픽셀이 200정도 이내에서 이동하도록 한다.
    for i in dlib_indexes('chin')[0:5] + dlib_indexes('chin')[-1:-5:-1]:
        const_y = variable_landmarks[i].y

        specify_row = edge_img[const_y, :]
        for diff in range(0, 201):
            target_x1 = variable_landmarks[i].x - diff
            target_x2 = variable_landmarks[i].x + diff

            if target_x1 >= 0:
                if specify_row[target_x1] == 255:
                    variable_landmarks[i] = dlib.point(target_x1, const_y)
                    break

            if target_x2 < img_width:
                if specify_row[target_x2] == 255:
                    variable_landmarks[i] = dlib.point(target_x2, const_y)
                    break

    return variable_landmarks


def redefine_eyebrow(landmarks):
    """
    redefine eyebrow landmark

    How to redefine eyebrow?
    -> About all eyebrow's landmark, each landmark cannot exceed the maximum x position among 0,1,2 index(about left eyebrow) or 14,15,16(about right eyebrow)
    -> if a x-position is exceeded, last landmark's x pos is redefined with x position of 0 index - 10 and redefined other landmarks x position to be equalliy distributed

    :param landmarks: landmarks(dlib point list)
    :return: landmarks(redefiend eyebrow)
    """
    cp_landmarks = copy.deepcopy(landmarks)

    range_left_eyebrow = dlib_detection_parser()['left eyebrow']
    range_right_eyebrow = dlib_detection_parser()['right eyebrow']

    left_eyebrow_indexes = [i for i in range(range_left_eyebrow[0], range_left_eyebrow[1] + 1)]
    right_eyebrow_indexes = [i for i in range(range_right_eyebrow[0], range_right_eyebrow[1] + 1)]

    left_eyebrow_start = left_eyebrow_indexes[0]
    left_eyebrow_end = left_eyebrow_indexes[-1]
    right_eyebrow_start = right_eyebrow_indexes[0]
    right_eyebrow_end = right_eyebrow_indexes[-1]

    left_face_end_x = cp_landmarks[0].x
    is_over_end_left = False
    for i in left_eyebrow_indexes:
        if cp_landmarks[i].x < left_face_end_x:
            is_over_end_left = True
            break

    if is_over_end_left:
        cp_landmarks[left_eyebrow_start].x = cp_landmarks[0].x - 10

        left_eyebrow_partition_x = partition_d1_2(cp_landmarks[left_eyebrow_start].x, cp_landmarks[left_eyebrow_end].x,
                                                  len(left_eyebrow_indexes) - 1)
        for i in left_eyebrow_indexes:
            value = int(left_eyebrow_partition_x[i - left_eyebrow_start])
            cp_landmarks[i].x = value

    right_face_end_x = cp_landmarks[16].x
    is_over_end_right = False
    for i in right_eyebrow_indexes:
        if right_face_end_x < cp_landmarks[i].x:
            is_over_end_right = True
            break

    if is_over_end_right:
        cp_landmarks[right_eyebrow_end].x = cp_landmarks[16].x - 10

        right_eyebrow_partition_x = partition_d1_2(cp_landmarks[right_eyebrow_start].x,
                                                   cp_landmarks[right_eyebrow_end].x, len(right_eyebrow_indexes) - 1)
        for i in right_eyebrow_indexes:
            value = int(right_eyebrow_partition_x[i - right_eyebrow_start])
            cp_landmarks[i].x = value

    return cp_landmarks


def landmark_detection_with_resizing(img, parser, detector, predictor, resize_h=320, resize_w=180, upsample_num=1):
    """
    This method detects landmarks of face. for being more faster, face image is resized and detect face landmark about the image
    and detected landmark is resized to be fitting the origin image

    320x180 is best resolution for dlib

    :param img: image
    :param parser: face parser
    :param detector: dlib detector
    :param predictor: dlib predictor
    :param resize_h: height of im
    :param resize_w: 이미지를 재조정할 폭
    :param upsample_num: It denotes that how many apply upsampling for detecting face landmark
    :return: landmarks
    """
    # landmark detection 속도를 높이기 위해 저해상도에서 얼굴 검출후 landmark detection 수행
    height, width = img.shape[0], img.shape[1]

    img_resize = cv2.resize(img, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    try:
        img_det = detector(img_resize, upsample_num)
        face_det = img_det[0]
    except:
        raise DlibFaceNotFoundException()

    x1, y1 = face_det.tl_corner().x, face_det.tl_corner().y
    x2, y2 = face_det.br_corner().x, face_det.br_corner().y

    height_ratio = height / resize_h
    width_ratio = width / resize_w

    x1_ = int(np.round(width_ratio * x1))
    x2_ = int(np.round(width_ratio * x2))
    y1_ = int(np.round(height_ratio * y1))
    y2_ = int(np.round(height_ratio * y2))

    try:
        shape = predictor(img, dlib.rectangle(x1_, y1_, x2_, y2_))
        img_landmarks = shape.parts()
    except:
        raise DlibLandmarkNotFoundException()

    # img_landmarks = redefine_face_border_landmarks(img, parser, img_landmarks)

    return img_landmarks


def landmark_detection(img, parser, detector, predictor, upsample_num=1):
    """
    detect face landmark

    :param img: image
    :param parser: face parser
    :param detector: dlib detector
    :param predictor: dlib predictor
    :param upsample_num: count of upsampling for detecting landmark
    :return: landmarks
    """
    try:
        img_dets = detector(img, upsample_num)
        if len(img_dets) == 0:
            raise Exception()
    except:
        raise DlibFaceNotFoundException()

    try:
        shape1 = predictor(img, img_dets[0])
        img_landmarks = shape1.parts()
    except:
        raise DlibLandmarkNotFoundException()

    # img_landmarks = redefine_face_border_landmarks(img, parser, img_landmarks)

    return img_landmarks


def warped_ref_landmarks(I_in_c_s, I_ref_c_s, I_in_align_landmarks):
    """
    warping landmark
    -> r_i^w = c_r + (s_i^a - c_s)
    -> r_i^w: warped landmark
    -> s_i^a: align landmark about input image
    -> c_s: center position of input image
    -> c_r: center position of reference image

    :param I_in_c_s: center position of input image
    :param I_ref_c_s: cetner position of reference image
    :param I_in_align_landmarks: input image's align landmark
    :return: warped landmarks(dlib point list)
    """
    warped_landmark_of_reference = []
    for landmark in I_in_align_landmarks:
        land_x = int(I_ref_c_s[0] + landmark.x - I_in_c_s[0])
        land_y = int(I_ref_c_s[1] + landmark.y - I_in_c_s[1])

        warped_landmark_of_reference.append([land_x, land_y])
        # 논문에서는 0, 16, 33의 landmark를 움직이지 않았는데, 이렇게하면 이미지가 깨져서 나와서 여기서는 움직였음
        # 왜냐하면, 특정한 좌표는 유지하면서 다른 좌표는 움직이니까 당연히 이미지가 이상하게 나올 수 밖에 없음
    return list(map(dlib.point, warped_landmark_of_reference))


def ref_landmark_warping(I_in_align_landmarks, I_ref_landmarks, I_in_c_s, I_ref_c_s):
    """
    get reference image's warped landmark

    :param I_in_align_landmarks: align input image's landmark
    :param I_ref_landmarks: reference image's landmark
    :param I_in_c_s: input image center position
    :param I_ref_c_s: reference imag center position
    :return: reference image's warped landmark
    """
    I_ref_warped_landmarks = warped_ref_landmarks(I_in_c_s, I_ref_c_s, I_in_align_landmarks)
    return I_ref_warped_landmarks


def detect_partialFaceArea_using_landmark(landmarks, partial_area):
    """
    Get face area

    :param landmarks: landmarks(dlib point list)
    :param partial_area: face area
    :return: face area(SJ_ellipse)
    """
    return_value = []
    if partial_area == 'chin':
        return_value = dlib_landmark_faceArea(landmark_pts=landmarks, face_area=partial_area)
    elif partial_area == 'left eyebrow':
        return_value = dlib_landmark_faceArea(landmark_pts=landmarks, face_area=partial_area)
    elif partial_area == 'right eyebrow':
        return_value = dlib_landmark_faceArea(landmark_pts=landmarks, face_area=partial_area)
    elif partial_area == 'upper nose':
        return_value = dlib_landmark_faceArea(landmark_pts=landmarks, face_area=partial_area)
    elif partial_area == 'side of nose':
        return_value = dlib_landmark_faceArea(landmark_pts=landmarks, face_area=partial_area)
    elif partial_area == 'left eye':
        left_eye_start = dlib_detection_parser()['left eye left_end']
        left_eye_end = dlib_detection_parser()['left eye right_end']

        center_x = int((landmarks[left_eye_start].x + landmarks[left_eye_end].x) / 2)
        center_y = int((landmarks[left_eye_start].y + landmarks[left_eye_end].y) / 2)

        x_radius = int(center_x - landmarks[left_eye_start].x)
        y_radius = x_radius

        return SJ_ellipse(center_pt=(center_x, center_y),
                          minor_radius=x_radius,
                          major_radius=y_radius)
    elif partial_area == 'right eye':
        right_eye_start = dlib_detection_parser()['right eye left_end']
        right_eye_end = dlib_detection_parser()['right eye right_end']

        center_x = int((landmarks[right_eye_start].x + landmarks[right_eye_end].x) / 2)
        center_y = int((landmarks[right_eye_start].y + landmarks[right_eye_end].y) / 2)

        x_radius = int(center_x - landmarks[right_eye_start].x)
        y_radius = x_radius

        return SJ_ellipse(center_pt=(center_x, center_y),
                          minor_radius=x_radius,
                          major_radius=y_radius)
    elif partial_area == 'lips outline':
        return_value = dlib_landmark_faceArea(landmark_pts=landmarks, face_area=partial_area)
    elif partial_area == 'lips inline':
        return_value = dlib_landmark_faceArea(landmark_pts=landmarks, face_area=partial_area)
    elif partial_area == 'right cheek':
        # find right cheek
        right_cheek_center_x = int((landmarks[54].x + landmarks[12].x) / 2)
        right_cheek_center_y = int((landmarks[29].y + landmarks[33].y) / 2)

        right_cheek_x_radius = landmarks[54].x - right_cheek_center_x
        right_cheek_y_radius = landmarks[33].y - right_cheek_center_y

        # find left cheek
        left_cheek_center_x = int((landmarks[4].x + landmarks[48].x) / 2)
        left_cheek_center_y = int((landmarks[29].y + landmarks[33].y) / 2)

        left_cheek_x_radius = left_cheek_center_x - landmarks[4].x
        left_cheek_y_radius = landmarks[33].y - left_cheek_center_y

        # choose bigger radius between right and left cheek
        biggest_radius = max(right_cheek_x_radius, right_cheek_y_radius, left_cheek_x_radius, left_cheek_y_radius)
        return_value = SJ_ellipse(center_pt=(right_cheek_center_x, right_cheek_center_y),
                                  minor_radius=biggest_radius,
                                  major_radius=biggest_radius)
    elif partial_area == 'left cheek':
        # find right cheek
        right_cheek_center_x = int((landmarks[54].x + landmarks[12].x) / 2)
        right_cheek_center_y = int((landmarks[29].y + landmarks[33].y) / 2)

        right_cheek_x_radius = landmarks[54].x - right_cheek_center_x
        right_cheek_y_radius = landmarks[33].y - right_cheek_center_y

        # find left cheek
        left_cheek_center_x = int((landmarks[4].x + landmarks[48].x) / 2)
        left_cheek_center_y = int((landmarks[29].y + landmarks[33].y) / 2)

        left_cheek_x_radius = left_cheek_center_x - landmarks[4].x
        left_cheek_y_radius = landmarks[33].y - left_cheek_center_y

        # choose the biggest radius between right and left cheek
        biggest_radius = max(right_cheek_x_radius, right_cheek_y_radius, left_cheek_x_radius, left_cheek_y_radius)
        return_value = SJ_ellipse(center_pt=(left_cheek_center_x, left_cheek_center_y),
                                  minor_radius=biggest_radius,
                                  major_radius=biggest_radius)
    elif partial_area == 'bellow left eye':
        center_x = int((landmarks[36].x + landmarks[39].x) / 2)
        center_y = int((landmarks[38].y + landmarks[40].y) / 2)

        supplementary_x_margin = int(abs(landmarks[36].x - landmarks[37].x) / 2)
        x_radius = int(center_x - landmarks[36].x)
        # left_eye_y_radius = int(abs(min(landmarks[40].y, landmarks[41].y) - center_y))

        diff_between_eye_and_nose_landmark = int(abs(min(landmarks[47].y, landmarks[46].y) - landmarks[28].y) / 2)
        y_radius = diff_between_eye_and_nose_landmark

        return_value = SJ_ellipse(center_pt=(center_x - int(supplementary_x_margin), center_y + y_radius),
                                  minor_radius=y_radius,
                                  major_radius=x_radius + supplementary_x_margin)
    elif partial_area == 'bellow right eye':
        center_x = int((landmarks[42].x + landmarks[45].x) / 2)
        center_y = int((landmarks[44].y + landmarks[46].y) / 2)

        supplementary_x_margin = int(abs(landmarks[44].x - landmarks[45].x) / 2)
        x_radius = int(center_x - landmarks[42].x)
        # right_eye_y_radius = int(abs(min(landmarks[47].y, landmarks[46].y) - center_y))

        diff_between_eye_and_nose_landmark = int(abs(min(landmarks[47].y, landmarks[46].y) - landmarks[28].y) / 2)
        y_radius = diff_between_eye_and_nose_landmark

        return_value = SJ_ellipse(center_pt=(center_x + int(supplementary_x_margin), center_y + y_radius),
                                  minor_radius=y_radius,
                                  major_radius=x_radius + supplementary_x_margin)
    elif partial_area == 'forehead':
        center_x = int((landmarks[21].x + landmarks[22].x) / 2)
        diff_eyebrow2nose = landmarks[28].y - landmarks[21].y
        center_y = landmarks[21].y - diff_eyebrow2nose

        x_radius = int(max(center_x - landmarks[17].x, center_x - landmarks[26].x))
        y_radius = int(max(landmarks[19].y - center_y, landmarks[24].y - center_y))

        return_value = SJ_ellipse(center_pt=(center_x, center_y),
                                  minor_radius=y_radius,
                                  major_radius=x_radius)
        return return_value
    elif partial_area == 'nasolabial folds':
        return_value = [(landmarks[5].x, landmarks[5].y), (landmarks[32].x, landmarks[32].y)]
    elif partial_area == 'galabella':
        return_value = 0
    elif partial_area == 'nose':
        items = dlib_landmark_faceArea(landmarks, 'upper nose')
        items = list(map(lambda e: [e.x, e.y], items))

        mean_pt = np.mean(items, axis=0, dtype=int)

        x_radius = int((mean_pt[0] - landmarks[39].x) / 2)
        y_radius = int(mean_pt[1] - landmarks[27].y)

        return_value = SJ_ellipse(center_pt=(int(mean_pt[0]), int(mean_pt[1])),
                                  minor_radius=x_radius,
                                  major_radius=y_radius)
    elif partial_area == 'side of left eye':
        eye_chin_dist = landmarks[36].x - landmarks[0].x  # 눈과 윤곽에 대한 거리

        x = (landmarks[0].x + landmarks[36].x) / 2 - int(eye_chin_dist / 6)
        y = landmarks[36].y

        x_radius = x - landmarks[0].x
        y_radius = y - landmarks[17].y

        return_value = SJ_ellipse(center_pt=(int(x), int(y)),
                                  minor_radius=int(x_radius / 2),
                                  major_radius=int(y_radius))
    elif partial_area == 'side of right eye':
        eye_chin_dist = landmarks[16].x - landmarks[45].x  # 눈과 윤곽에 대한 거리

        x = (landmarks[16].x + landmarks[45].x) / 2 + int(eye_chin_dist / 6)
        y = landmarks[45].y

        x_radius = landmarks[16].x - x
        y_radius = y - landmarks[26].y

        return_value = SJ_ellipse(center_pt=(int(x), int(y)),
                                  minor_radius=int(x_radius / 2),
                                  major_radius=int(y_radius))

    return return_value


def redefine_drawing_img(origin_img, origin_landmarks, drawing_img, parser):
    """
    redefine drawing image

    :param origin_img: origin image
    :param origin_landmarks: origin image's landmark
    :param drawing_img: closed line image
    :param parser: face parser
    :return: closed line image(it deletes partial areas that is lips and eye)
    """
    img_height, img_width, channel = origin_img.shape[0], origin_img.shape[1], origin_img.shape[2]
    mask = np.zeros((img_height, img_width, channel), dtype=np.uint8)

    face_parsing_indexes = face_parsing_index()
    lip_area = find_faceRoi(parser, origin_img, [face_parsing_indexes['mouth'], face_parsing_indexes['upper lip'],
                                                 face_parsing_indexes['lower lip']])
    lip_mask = np.array(np.where(lip_area > 0, 255, 0), dtype=np.uint8)
    mask = cv2.add(mask, lip_mask)

    eyebrow_area = find_faceRoi(parser, origin_img,
                                [face_parsing_indexes['left eyebrow'], face_parsing_indexes['right eyebrow']])
    eyebrow_mask = np.array(np.where(eyebrow_area > 0, 255, 0), dtype=np.uint8)
    mask = cv2.add(mask, eyebrow_mask)

    left_eye_circle = detect_partialFaceArea_using_landmark(origin_landmarks, 'left eye')
    left_eye_mask = np.zeros((img_height, img_width, channel), dtype=np.uint8)
    cv2.ellipse(img=left_eye_mask,
                center=left_eye_circle._center_pt,
                axes=(left_eye_circle._major_radius, int(left_eye_circle._major_radius / 2)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=-1)
    mask = cv2.add(mask, left_eye_mask)

    right_eye_circle = detect_partialFaceArea_using_landmark(origin_landmarks, 'right eye')
    right_eye_mask = np.zeros((img_height, img_width, channel), dtype=np.uint8)
    cv2.ellipse(img=right_eye_mask,
                center=right_eye_circle._center_pt,
                axes=(right_eye_circle._major_radius, int(right_eye_circle._major_radius / 2)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=-1)
    mask = cv2.add(mask, right_eye_mask)

    return np.where(mask == 255, 0, drawing_img)
