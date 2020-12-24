'''
https://github.com/shaoanlu/face_toolbox_keras
'''
import numpy as np

def find_faceRoi_withParser(parser, img):
    """
    find face roi

    :param parser: face parser
    :param img: image
    :return: image about face roi
    """
    interest_face_index = [1,2,3,4,5,6,10,11,12,13]

    return find_faceRoi(parser, img, interest_face_index)

def find_faceRoi(parser, img, interests):
    """
    find face roi

    :param parser: face parser
    :param img: image
    :param interests: interest areas ex) [1,2,3,4,5,6,10,11,12,13]
    :return: image about face roi
    """
    out = parser.parse_face(img)

    face_only = np.logical_not(np.isin(out[0], interests))
    face_only = np.expand_dims(face_only, axis=0).reshape(face_only.shape[0], face_only.shape[1], 1)
    face_mask = np.concatenate([face_only, face_only, face_only], axis=2)

    img_roi = np.ma.masked_array(img, mask=face_mask, fill_value=0).filled()

    return img_roi

def face_parsing_index():
    """
    face parsing indexes

    :return: dictionary: { description: index }
    """
    # https://github.com/shaoanlu/face_toolbox_keras/blob/master/demo.ipynb
    return {
        'background' : 0,
        'skin' : 1,
        'left eyebrow' : 2,
        'right eyebrow' : 3,
        'left eye' : 4,
        'right eye' : 5,
        'glasses' : 6,
        'left ear' : 7,
        'right ear' : 8,
        'earnings' : 9,
        'nose' : 10,
        'mouth' : 11,
        'upper lip' : 12,
        'lower lip' : 13,
        'neck' : 14,
        'neck_l' : 15,
        'cloth' : 16,
        'hair' : 17,
        'hat' : 18,
    }
