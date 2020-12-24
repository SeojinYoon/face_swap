# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:10:52 2020

@author: Seojin
"""


def face_swap(in_path, ref_path):
    print("input path: ", in_path)
    print("ref path: ", ref_path)
    import tensorflow as tf
    import keras

    if ((tf.__version__ == '1.12.0' or tf.__version__ == '1.13.1') and keras.__version__ == '2.2.4') != True:
        # https://github.com/shaoanlu/face_toolbox_keras
        # requirements:
        # Keras: 2.2.4
        # Tensorflow: 1.12.0 or 1.13.1
        raise Exception('Please match tensorflow and keras version. tf needs 1.12.0 or 1.13.1 and keras needs 2.2.4')

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

    _, I_in2, _, I_in2_landmarks = main_face_align(I_in=I_ref,
                                                   I_ref=I_in,
                                                   I_in_landmarks=I_ref_landmarks,
                                                   I_ref_landmarks=I_in_landmarks,
                                                   detector=detector,
                                                   predictor=predictor)

    I_in_align, I_ref2, _, I_ref2_landmarks = main_face_align(I_in=I_in2,
                                                              I_ref=I_ref,
                                                              I_in_landmarks=I_in2_landmarks,
                                                              I_ref_landmarks=I_ref_landmarks,
                                                              detector=detector,
                                                              predictor=predictor)

    result = face_replacement(I_in_align, I_ref2, I_in2_landmarks, I_ref2_landmarks, parser, True)

    return result

















