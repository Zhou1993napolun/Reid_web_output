# coding=utf-8

# Plot without display
# must put before using any display backend

from io import BytesIO
import time

from cam_test import MJPEGClient
import http.client
from urllib import parse

import threading
from flask import Flask, render_template, Response

import os
import cv2
import torch
import warnings
import argparse
import numpy as np
import onnxruntime as ort
from utils.datasets1 import LoadStreams, LoadImages
from utils.draw import draw_boxes
from utils.general import check_img_size
from utils.torch_utils import time_synchronized
from person_detect_yolov5 import Person_detect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized

import math
from PIL import Image, ImageDraw, ImageFont


FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 0))  # second
#RES_URL = os.getenv('RES_URL', "http://192.168.27.210:8080/video_feed")
# RES_URL = os.getenv('RES_URL', "http://192.168.12.150:8090/stream.mjpg")
RES_URL = os.getenv('RES_URL', "http://0.0.0.0:8080/index.html")

ORION_TASK_ID = os.getenv('ORION_TASK_ID', "test_0001")
OBJECT_OUTPUT_PORT = os.getenv('OBJECT_OUTPUT_PORT', "8080")

batchsize = 1 # Can only be 1


inputFrame = None
wait_time = 2

outputFrame = None
lock = threading.Lock()
loginfo = "Video Stream is Running..."

colors = np.array([
    [1,0,1],
    [0,0,1],
    [0,1,1],
    [0,1,0],
    [1,1,0],
    [1,0,0]
    ])


app = Flask(__name__)

@app.route('/')
def index():
    global loginfo
    """Video streaming home page."""
    return render_template('index.html', loginfo = loginfo)


def get_frame():
    global inputFrame, loginfo
    while True:
        try:
            print("*****************")
            print("I have tried")
            for jpegdata in MJPEGClient(RES_URL):
                print("jeegdata is : ")
                print(jpegdata)
            for jpegdata in MJPEGClient(RES_URL):
                print("InputFrame is updated")
                response = BytesIO(jpegdata)
                img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
                inputFrame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        except Exception as e:
            loginfo = 'error, %s. \r\n\r\n Try again in %s seconds.' % (e, str(wait_time))
            print(loginfo)
            loginfo = ORION_TASK_ID + ' @@ ' + loginfo
            time.sleep(wait_time)
            pass


# real time data deal
def frame_pre_deal(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



# real time data deal

# reid detection part

def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format
    return midpoint

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))

def get_size_with_pil(label,size=25):
    font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8")  # simhei.ttf
    return font.getsize(label)

def put_text_to_cv2_img_with_pil(cv2_img,label,pt,color):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2???PIL????????????hex?????????????????????????????????RGB??????
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()???????????????????????????????????????np.array()??????
    draw = ImageDraw.Draw(pilimg)  # PIL?????????????????????
    font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8") #simhei.ttf
    draw.text(pt, label, color,font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # ???????????????cv2.imshow()???????????????????????????

def get_color(c, x, max):
    ratio = (x / max) * 5;
    i = math.floor(ratio);
    j = math.ceil(ratio);
    ratio -= i;
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    return r;

def compute_color_for_labels(class_id,class_total=80):
    offset = (class_id + 0) * 123457 % class_total;
    red = get_color(2, offset, class_total);
    green = get_color(1, offset, class_total);
    blue = get_color(0, offset, class_total);
    return (int(red*256),int(green*256),int(blue*256))

class yolo_reid():
    def __init__(self, cfg, args, camera):
        self.logger = get_logger("root")
        self.args = args
        # self.video_path = path
        self.camera = camera
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        # self.person_detect = Person_detect(self.args, self.video_path)
        self.person_detect = Person_detect(self.args, self.camera)
        imgsz = check_img_size(args.img_size, s=32)  # self.model.stride.max())  # check img_size
        # self.dataset = LoadImages(self.video_path, img_size=imgsz)

        self.dataset = LoadStreams(self.camera, img_size=imgsz)
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)
        self.img_cnt = 0

    def deep_sort(self):

        global inputFrame, outputFrame, lock, loginfo

        idx_frame = 0

        image_lists = []
        while False:
            try:
                print("$$$$$$$$$$$$")
                print("input Frame is : ")
                print(inputFrame)
                if inputFrame is None:
                    continue
                image_lists.append(inputFrame)
                ori_img = image_lists[0]

                # Padded resize
                img = frame_pre_deal(ori_img, new_shape=640)[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                video_path = None
                vid_cap = None

                bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)
                outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

                # ????????????
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_img = draw_boxes(ori_img, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                cv2.imwrite("./temp_img/{}.jpg".format(self.img_cnt), ori_img)
                self.img_cnt += 1
                print(self.img_cnt)

                # add for http video
                flag, encodedImage = cv2.imencode(".jpg", ori_img)

                my_stringIObytes = BytesIO(encodedImage)
                my_stringIObytes.seek(0)

                with lock:
                    outputFrame = my_stringIObytes.read()

                # add for http video

            except Exception as e:
                loginfo = 'error, %s. \r\n\r\n Try again in %s seconds.' % (e, str(wait_time))
                print(loginfo)
                loginfo = ORION_TASK_ID + ' @@ ' + loginfo
                time.sleep(wait_time)
                pass

        if True:
            for video_path, img, ori_img, vid_cap in self.dataset:
                idx_frame += 1
                # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
                t1 = time_synchronized()

                # yolo detection
                bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)

                cv2.imwrite("./12_1_temp_img/test.jpg", ori_img[0])

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_img)


                # The "outputs" is a list object
                # The outputs[0] is like this [x1 y1 x2 y2 ID] and the x1 y1 x2 y2 is the boxes location
                # print(len(outputs))
                # if len(outputs) > 0:
                #     print("The output shape is : ")
                #     print(outputs[0].shape)
                #     print(outputs)

                # 3. ????????????
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_img = draw_boxes(ori_img, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                print("yolo+deepsort:", time_synchronized() - t1)

                end = time_synchronized()

                print("*****************")
                print(type(ori_img))
                print(len(ori_img))
                print(type(ori_img[0]))

                # add for http video
                flag, encodedImage = cv2.imencode(".jpg", ori_img[0])

                my_stringIObytes = BytesIO(encodedImage)
                my_stringIObytes.seek(0)

                with lock:
                    outputFrame = my_stringIObytes.read()

                # add for http video

                if self.args.display:
                    # cv2.imshow("test", ori_img)
                    # cv2.imwrite("./temp_img/{}.jpg".format(self.img_cnt), ori_img)
                    self.img_cnt += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.logger.info("{}/time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                                 .format(idx_frame, end - t1, 1 / (end - t1),
                                         bbox_xywh.shape[0], len(outputs)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='./tc.mp4', type=str)
    # parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--camera", dest="cam", default='/dev/video0',type=str)
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=True, help='True: sort model, False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    return parser.parse_args()

def flask_reid():
    global inputFrame

    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    my_yolo_reid = yolo_reid(cfg, args, camera=args.cam)

    with torch.no_grad():
        my_yolo_reid.deep_sort()

# reid detection part

def generate():
    """Video streaming generator function."""
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            if outputFrame is None:
                continue
            encodedImage = outputFrame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # start a thread that will get frames
    # t1 = threading.Thread(target=get_frame)
    # t1.daemon = True
    # t1.start()
    #
    # # wait 3 seconds to get the frames ready
    # time.sleep(3)

    # start a thread that will perform motion detection
    # t = threading.Thread(target=object_detection,args = (infer,))
    t = threading.Thread(target=flask_reid)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', threaded=True, debug=True, port="8080", use_reloader=False)
