import cv2
import numpy as np
import torch
# img = cv2.imread('./final_data_img/0.png')
#
# img_0 = cv2.resize(img, (56, 56))
#
# cv2.imwrite('test_out.png', img_0)

# a = torch.tensor([[[3, 2, 1], [1, 2, 3], [2, 3, 1]], [[3, 2, 1], [1, 2, 3], [2, 3, 1]]])
# print(a.shape)
# print(a[:, 0])
# b = a[:, 0] == 1
# c = a[b, :]
# print(c)

import cv2

img_num = 300

image_ori = cv2.imread("./class_temp_img/0.jpg")

video_size = (image_ori.shape[1], image_ori.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("./self_work/output_only_clo.mp4",  fourcc, 30, video_size, True)

for i in range(img_num):
    print(i)
    frame = cv2.imread("./class_temp_img/{}.jpg".format(i))
    video.write(frame)
video.release()

print("查看生成的视频")