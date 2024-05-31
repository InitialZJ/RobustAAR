# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

# 当前py文件所在的目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements,
                           colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import PX4MavCtrlV4 as PX4MavCtrl

import time
from queue import Queue
import threading

# 准备向simulink中传输数据
mav = PX4MavCtrl.PX4MavCtrler(20100)
mav.initUE4MsgRec()

# def InitialPos():
# simulink中设定
PosWorldTanker = np.array([260, 8, -1320])
PosCenterToModel = np.array([0, 0, 0])
# PosRelativeReceiverToTanker = [-40, -1, 14] + PosCenterToModel
PosRelativeReceiverToTanker = np.array([-40, -1, 14])
PosWorldReceiver = PosWorldTanker + PosRelativeReceiverToTanker
# json文件中设定
PosRelativeCameraToReceiver = np.array([4.7, 0.54, -1.45]) - np.array([0, 0, 2.05])
# 实际上下面两个参数并未使用，在Rflysim3D中需要校正
PosRelativeProbeToReceiver = np.array([7.3712, 0.5497, -1.4418])
PosRelativeNoseToReceiver = np.array([9.6940, 0, 0])
# 锥套相关参数
PosOilinToDrogue = np.array([0.347, 0, 0])
PosUmbrelaaToDrogue = np.array([1.3210, 0, 0])
# 这里有过放缩 是之前数据的2倍
PosRelativeDrogueToTanker = np.array([-27.08, -0.142, 9.7862])
# 加油机初始距离
initial_dis = 90
d1 = 90
d2 = 40
send_flag = False


class RecordPosAndDistance:
    def __init__(self):
        # 使用前3个frame
        self.frame = 10
        self.N = 3
        self.distance = [65.535] * self.frame
        self.center_x = [0] * self.frame
        self.center_y = [0] * self.frame
        self.estimate_r = [0] * self.frame

        self.mean_center_x = [0] * (self.frame - 1)
        self.mean_center_y = [0] * (self.frame - 1)
        self.mean_estimate_r = [0] * (self.frame - 1)

        self.std_center_x = [0] * (self.frame - 1)
        self.std_center_y = [0] * (self.frame - 1)
        self.std_estimate_r = [0] * (self.frame - 1)

    def updateMeanStd(self):
        # 每次更新一下这几个参数
        self.mean_center_x.append(np.mean(self.center_x[-self.frame:]))
        self.std_center_x.append(np.std(self.center_x[-self.frame:]))
        self.mean_center_y.append(np.mean(self.center_y[-self.frame:]))
        self.std_center_y.append(np.std(self.center_y[-self.frame:]))
        self.mean_estimate_r.append(np.mean(self.estimate_r[-self.frame:]))
        self.std_estimate_r.append(np.std(self.estimate_r[-self.frame:]))

    def insertZc(self, zc):
        # 对Zc的值进行控制
        if zc <= 2.67 or zc >= d2:
            self.distance.append(self.distance[-1])
        else:
            self.distance.append(zc)
        return self.distance[-1]

    def insertNewXYR(self, new_x, new_y, new_r):
        # 满足一定条件，才插入
        if len(self.center_x) <= 2 * self.frame:
            self.center_x.append(new_x)
            self.center_y.append(new_y)
            self.estimate_r.append(new_r)
        else:
            # 自己设定偏差的百分比
            rate = 0.5
            if self.mean_center_x[-1] * (1 - rate) <= new_x <= self.mean_center_x[-1] * (1 + rate) \
                    and self.mean_center_y[-1] * (1 - rate) <= new_y <= self.mean_center_y[-1] * (
                    1 + rate) \
                    and self.mean_estimate_r[-1] * (1 - rate) <= new_r <= self.mean_estimate_r[
                -1] * (1 + rate):
                self.center_x.append(new_x)
                self.center_y.append(new_y)
                self.estimate_r.append(new_r)
            else:
                self.center_x.append(self.center_x[-1])
                self.center_y.append(self.center_y[-1])
                self.estimate_r.append(self.estimate_r[-1])

    def printDataX(self):
        print("center_x:", self.center_x)
        print("mean_center_x:", self.mean_center_x)
        print("std_center_x:", self.std_center_x)

    def printDataY(self):
        print("center_y:", self.center_y)
        print("mean_center_y:", self.mean_center_y)
        print("std_center_y:", self.std_center_y)

    def printDataR(self):
        print("estimate_r:", self.estimate_r)
        print("mean_estimate_r:", self.mean_estimate_r)
        print("std_estimate_r:", self.std_estimate_r)

    def printAll(self):
        self.printDataX()
        self.printDataY()
        self.printDataR()


# 滤去不合适的数据
recordPosAndDistance = RecordPosAndDistance()


# Zc是深度相机测量出来的距离、
def cal_pos_by_camera_matrix(u, v, Zc):
    PixelCorCenterX = 639.69896228
    PixelCorCenterY = 357.42416711
    fx = 653.39324951
    fy = 634.93518066
    deltaX = Zc * (u - PixelCorCenterX) / fx
    deltaY = Zc * (v - PixelCorCenterY) / fy
    # 返回值的坐标系是世界坐标系，上面计算出来的是相机坐标系
    return [Zc, deltaX, deltaY]


def estimateDrogue(distance, center_x, center_y, recordPosAndDistance):
    global send_flag
    Zc = distance + 0.7
    print("distance:", Zc)
    estimatedroguePos = cal_pos_by_camera_matrix(center_x, center_y, Zc)
    return estimatedroguePos


def generateXYZ(t):
    x = 6 * math.sin(0.5 * t)
    y = math.sin(2.5 * 0.6 * t + math.pi)
    z = math.sin(2.5 * t)
    return [x - 11, y - 1, z + 1.3]


@torch.no_grad()
def run(weights=ROOT / 'best.pt',  # model.pt path(s)
        # source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        source='',
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        # max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        # add
        record='',
        ):
    save_txt = True
    hide_labels = False

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run save-dir=run/detect/exp
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half   在半精度设置下 pt=1 且使用GPU训练的时候 才使用半精度进行训练
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = len(dataset)  # batch_size

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(
            torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    w = 1280
    h = 720
    dis = initial_dis

    t = 0
    t_total = 14
    li_x, li_y, li_z = [], [], []
    for path, im, im0s, vid_cap, s in dataset:
        x, y, z = generateXYZ(t)

        simulink_data = mav.inSilVect[0].inSILFLoats[:6]
        pos_drogue = simulink_data[:3]

        mav.sendUE4Pos(1, 213, PosE=[pos_drogue[0] + x - PosRelativeCameraToReceiver[0],
                                     pos_drogue[1] + y - PosRelativeCameraToReceiver[1],
                                     pos_drogue[2] + z - PosRelativeCameraToReceiver[2]])
        time.sleep(0.5)

        path = "1"
        p, im0, frame = path, np.copy(im0s), getattr(dataset, 'frame', 0)
        im0 = im0[0]
        img_depth = dataset.imgs_d[0]
        h, w = img_depth.shape

        img_depth = np.where(img_depth <= 3650, 65535, img_depth)
        min_val = np.min(img_depth)
        depth_coordinates = np.where((img_depth >= min_val) & (img_depth <= min_val + 600))

        print(len(depth_coordinates[0]))
        center_x, center_y, radius = -1, -1, -1
        if len(depth_coordinates[0]) >= 100:
            depth_box = [[np.min(depth_coordinates[1]), np.min(depth_coordinates[0])],
                         [np.max(depth_coordinates[1]), np.max(depth_coordinates[0])]]

            expand = 10
            bgr_box = [[max(depth_box[0][0] - expand, 0),
                        max(depth_box[0][1] - expand, 0)],
                       [min(depth_box[1][0] + expand, w - 1),
                        min(depth_box[1][1] + expand, h - 1)]]

            img_sleeve = im0[bgr_box[0][1]: bgr_box[1][1], bgr_box[0][0]: bgr_box[1][0], :]
            cv2.imshow("img_sleeve", img_sleeve)
            img_gray = cv2.cvtColor(img_sleeve, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.medianBlur(img_gray, 5)
            _, img_binary = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)

            circles = cv2.HoughCircles(img_binary, cv2.HOUGH_GRADIENT_ALT, 1, 15,
                                       param1=300, param2=0.8, minRadius=0, maxRadius=0)
            print(circles)

            if circles is None:
                # 无法检测到完整圆形，对锥套进行圆形拟合
                c = np.array(depth_coordinates, dtype=np.int32).transpose((1, 0))
                c[:, [0, 1]] = c[:, [1, 0]]
                c = np.expand_dims(c, axis=1)

                (center_x, center_y), radius = cv2.minEnclosingCircle(c)
                # 转为整数 cast to integers
                center_x, center_y, radius = int(center_x), int(center_y), int(radius)
            else:
                # 能检测到完整的圆形的情况
                circles = np.uint16(np.around(circles))
                radius = 0
                for i in circles[0, :]:
                    if i[2] > radius:
                        center_x, center_y, radius = i[0] + bgr_box[0][0], i[1] + bgr_box[0][1], i[2]

        if center_x == -1:
            print("Wrong")
            break
        x_show = center_x
        y_show = center_y
        r_show = radius

        cv2.circle(im0, (x_show, y_show), r_show, (0, 255, 0), 2)
        size = 20
        left = int(x_show - 0.5 * size)
        right = int(x_show + 0.5 * size)
        up = int(y_show - 0.5 * size)
        down = int(y_show + 0.5 * size)
        cv2.line(im0, (left, y_show), (right, y_show), (0, 255, 0), 1)
        cv2.line(im0, (x_show, up), (x_show, down), (0, 255, 0), 1)

        print("x_show", x_show)
        print("y_show", y_show)
        # 根据rgb图像的中心点坐标和深度图像来计算距离，进而估算锥套位置  把这部分拆除出去，不要影响取图速度
        res = estimateDrogue(float(min_val / 1000.0), x_show, y_show, recordPosAndDistance)
        cv2.imshow("im0_rgb", im0)
        cv2.imshow('img_depth', img_depth)
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        li_x.append(-res[0])
        li_y.append(-res[1])
        li_z.append(-res[2])
        t += 0.2
        if t > t_total:
            csv_file = 'data.csv'
            data = zip(li_x, li_y, li_z)
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x', 'y', 'z'])  # 写入 CSV 文件的标题行
                writer.writerows(data)
            print('Finish')
            break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'runs/train/exp14/weights/best.pt',
                        help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default='', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[720, 1280],
    #                     help='inference size h,w')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 保存我从UE4中取到的视频的地方
    parser.add_argument('--record',
                        default=r"D:\project\空加平台\AAR平台整理3.21\Python视觉处理\runs\record\1.avi",
                        action='store_true',
                        help='save picture from UE4')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    print("start!")
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
