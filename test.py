
from face_swap import face_swap
import os
import matplotlib.pylab as plt

man_path = os.path.join(".", "test_image", "woman.jpg")
woman_path = os.path.join(".", "test_image", "man.jpg")

img = face_swap(man_path, woman_path)

plt.imshow(img)






in_path = man_path
ref_path = woman_path


import tensorflow as tf
import keras

import dlib
import matplotlib.pylab as plt
import cv2

from components.sj_image_process import upright_camera_img, sizing_aspect_ratio
from components.landmark_detector import landmark_detection_with_resizing, landmark_scaling

from face_toolbox_keras_master.models.parser import face_parser

from components.face_process import main_face_align, face_replacement

parser = face_parser.FaceParser()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

'''
reference paper:
    https://www.hindawi.com/journals/mpe/2019/8902701/

reference source:
    https://github.com/BruceMacD/Face-Swap-OpenCV
    https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/

note:
    This algorithm is based on dlib's landmark detection so if dlib cannot detect landmark, cannot swap face 

dependency:
    dlib: http://dlib.net/
    face_toolbox_keras: https://github.com/shaoanlu/face_toolbox_keras
'''

I_ref = plt.imread(ref_path)
(I_ref_resize_w, I_ref_resize_h) = sizing_aspect_ratio(I_ref.shape[0], I_ref.shape[1],
                                                       height=320)  # 얼굴 비율에 맞지 않도록 이미지를 축소시키면 landmark detection이 안되는 경우가 존재함
try:
    I_ref_landmarks = landmark_detection_with_resizing(I_ref, parser, detector, predictor, resize_h=I_ref_resize_h,
                                                       resize_w=I_ref_resize_w)
except Exception as ex:
    print('landmark is not detected in I_ref')

I_in = plt.imread(in_path)
I_in_h, I_in_w = I_in.shape[0], I_in.shape[1]
(I_in_resize_w, I_in_resize_h) = sizing_aspect_ratio(I_in.shape[0], I_in.shape[1], height=320)
try:
    I_in_landmarks = landmark_detection_with_resizing(I_in, parser, detector, predictor, resize_h=I_in_resize_h,
                                                      resize_w=I_in_resize_w)
except Exception as ex:
    print('landmark is not detected in I_in')
I_in = cv2.resize(plt.imread(in_path), dsize=(I_ref.shape[1], I_ref.shape[0]),
                  interpolation=cv2.INTER_AREA)  # 이미지 연산을 위해 shape을 맞춰줌
I_in_landmarks = landmark_scaling(landmarks=I_in_landmarks, x_scale=I_ref.shape[1] / I_in_w,
                                  y_scale=I_ref.shape[0] / I_in_h)

from components.face_process import find_center_pt, average_distance, diff_eye_angle, img_warp_top_to_face, img_warp_bottom_to_face, warping_triangle
from components.sj_image_process import Image_preprocessing
import numpy as np
from components.landmark_detector import ref_landmark_warping
from components.landmark_detector import dlib_indexes, dlib_landmark_range
from components.sj_util import get_multiple_elements_in_list, partition_d1
import scipy

I_in_c_s = find_center_pt(I_in_landmarks)
I_ref_c_s = find_center_pt(I_ref_landmarks)

desired_face_measure = average_distance(I_ref_landmarks, I_ref_c_s)
input_face_measure = average_distance(I_in_landmarks, I_in_c_s)
k = desired_face_measure / input_face_measure

rot_angle = diff_eye_angle(I_in_landmarks, I_ref_landmarks)

# image scaling, face rotating
print("diff angle 2: ", rot_angle)
print("scale value is: ", k)

if rot_angle > 0:
    I_in_rot = Image_preprocessing.rotate_center(I_in, np.abs(rot_angle), I_in_c_s, True, k)
else:
    I_in_rot = Image_preprocessing.rotate_center(I_in, np.abs(rot_angle), I_in_c_s, False, k)

I_in_align = I_in_rot

I_in_align_landmarks = Image_preprocessing.rotate_point(list(map(lambda x: [x.x, x.y], I_in_landmarks)), I_in_c_s,
                                                        rot_angle, k)
I_in_align_landmarks = list(map(lambda x: dlib.point(x[0], x[1]), I_in_align_landmarks))

I_ref_warped_landmarks = ref_landmark_warping(I_in_align_landmarks, I_ref_landmarks, I_in_c_s, I_ref_c_s)


face_indexs = [i for i in range(dlib_landmark_range()[0], dlib_landmark_range()[1] + 1)]

img_src_h, img_src_w = I_ref.shape[0], I_ref.shape[1]

chin_indexes = dlib_indexes('chin')
upper_face_indexes = [chin_indexes[0]] + dlib_indexes('left eyebrow') + dlib_indexes('right eyebrow') + [
    chin_indexes[-1]]

list_I_ref_landmarks = np.array(
    list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_landmarks, face_indexs))))
list_I_ref_warped_landmarks = np.array(
    list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_warped_landmarks, face_indexs))))

# 이미지의 맨위에 있는 line을 분할
partition_pts = list(map(lambda x: [x, 0], [int(pt[0]) for pt in partition_d1(0, img_src_w, 10)] + [img_src_w - 1]))

# 눈 주위에 있는 필요한 landmark 뽑아내기
I_ref_eyebrow_landmarks = get_multiple_elements_in_list(list_I_ref_landmarks.tolist(), upper_face_indexes)
I_ref_warp_eyebrow_landmarks = get_multiple_elements_in_list(list_I_ref_warped_landmarks.tolist(),
                                                             upper_face_indexes)

# 분할된 점과 landmark를 통합
I_ref_background_pts = I_ref_eyebrow_landmarks + partition_pts
I_warped_ref_background_pts = I_ref_warp_eyebrow_landmarks + partition_pts

# reference image, warped_reference image 분할
triangles = scipy.spatial.Delaunay(I_ref_background_pts)
delaunay_background_indexes = triangles.simplices

delaunay_index_map = {
    'first': 0,
    'last': 11
}
delaunay_background_indexes = np.array(
    list(filter(lambda x: sum(np.isin(x, range(delaunay_index_map['first'], delaunay_index_map['last'] + 1))) != 3,
                delaunay_background_indexes)))

I_ref_background_triangle_pts = np.array(I_ref_background_pts)[delaunay_background_indexes]
I_warped_ref_background_triangle_pts = np.array(I_warped_ref_background_pts)[delaunay_background_indexes]

# Warping Process
results = []
for i in range(0, len(I_ref_background_triangle_pts)):
    ref_triangle_pt = I_ref_background_triangle_pts[i]
    warped_ref_triangle_pt = I_warped_ref_background_triangle_pts[i]

    res = warping_triangle(ref_triangle_pt, warped_ref_triangle_pt, I_ref)
    results.append(res)

I_ref_cover = np.zeros((img_src_h, img_src_w, 3), np.uint8)
for warp_triangle in results:
    I_ref_cover = cv2.add(I_ref_cover, warp_triangle)




face_indexs = [i for i in range(dlib_landmark_range()[0], dlib_landmark_range()[1] + 1)]

img_src_h, img_src_w = I_ref.shape[0], I_ref.shape[1]

list_I_ref_landmarks = np.array(
    list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_landmarks, face_indexs))))
list_I_ref_warped_landmarks = np.array(
    list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_warped_landmarks, face_indexs))))

# 들로네 삼각 분할을 적용하여 얼굴 영역 분할
triangles = scipy.spatial.Delaunay(
    list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_landmarks, face_indexs))))

delaunay_trianglePt_indexes = triangles.simplices

from components.face_process import triangle_outter_pts
I_ref_triangles = list_I_ref_landmarks[delaunay_trianglePt_indexes].tolist()
I_warp_triangles = list_I_ref_warped_landmarks[delaunay_trianglePt_indexes].tolist()
# 얼굴이외의 배경 영역을 warping하기 위하여 삼각형 추가
I_ref_triangles += triangle_outter_pts(I_ref_landmarks, img_src_h, img_src_w)
I_warp_triangles += triangle_outter_pts(I_ref_warped_landmarks, img_src_h, img_src_w)

I_ref_background_triangle_pts.shape
I_warped_ref_background_triangle_pts.shape
np.array(I_ref_triangles).shape
np.array(I_warp_triangles).shape
from components.face_process import triangle_outter_pts

tr1 = np.concatenate((I_ref_background_triangle_pts, np.array(I_ref_triangles)))
tr2 = np.concatenate((I_warped_ref_background_triangle_pts, np.array(I_warp_triangles)))

from components.draw_util import draw_traingles
plt.imshow(draw_traingles(tr1, I_ref, 2))
plt.imshow(draw_traingles(tr2, I_in, 2))


results = []
for i in range(0, len(I_ref_triangles)):
    ref_triangle_pt = I_ref_triangles[i]
    warped_ref_triangle_pt = I_warp_triangles[i]

    res = warping_triangle(ref_triangle_pt, warped_ref_triangle_pt, I_ref)
    results.append(res)

I_ref_warp = np.zeros((img_src_h, img_src_w, 3), np.uint8)
for warp_triangle in results:
    # warping된 삼각형에서 이미 픽셀이 채워진 영역은 필요없으므로 삭제
    warped_triangle = remove_overlapped_pixels(warp_triangle, I_ref_warp)
    I_ref_warp = cv2.add(I_ref_warp, warped_triangle)
