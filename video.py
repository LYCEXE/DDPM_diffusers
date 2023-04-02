import cv2
import numpy as np

size = (1024, 512)

video_write = cv2.VideoWriter(r'celeba_train_2/train_2/video.mp4',0x7634706d,5,size)
img_array = []
for filename in [r"celeba_train_2/train_2/samples/{:04d}.png".format(i) for i in range(100)]:
    img = cv2.imread(filename)
    if img is None:
        print(filename + " is error!")
        continue
    img_array.append(img)
for i in range(100):
    video_write.write(img_array[i])
video_write.release()
