# Face_swap

## How to use it?

from face_swap import face_swap
import os
import matplotlib.pylab as plt

man_path = os.path.join(".", "test_image", "woman.jpg")
woman_path = os.path.join(".", "test_image", "man.jpg")

img = face_swap(man_path, woman_path)

plt.imshow(img)

## Algorithm

![ex_screenshot](./face_swap algorithm_설명(1).png)

## Reference

I refer to the paper and sources

paper: https://www.hindawi.com/journals/mpe/2019/8902701/
source: https://github.com/BruceMacD/Face-Swap-OpenCV
	https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/

## Dependency

dlib: http://dlib.net/
face_toolbox_keras: https://github.com/shaoanlu/face_toolbox_keras
