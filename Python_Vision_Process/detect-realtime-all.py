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
PosRelativeCameraToReceiver = np.array([4.7, 0.54, -1.45])
# 实际上下面两个参数并未使用，在Rflysim3D中需要校正
PosRelativeProbeToReceiver = np.array([7.3712, 0.5497, -1.4418])
PosRelativeNoseToReceiver = np.array([9.6940, 0, 0])
# 锥套相关参数
PosOilinToDrogue = np.array([0.347, 0, 0])
PosUmbrelaaToDrogue = np.array([1.3210, 0, 0])
# 这里有过放缩 是之前数据的2倍
PosRelativeDrogueToTanker = np.array([-27.08, -0.142, 9.7862])
# 加油机初始距离
initial_dis = 130
d1 = 90
d2 = 40


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


def cal_pixel_by_camera_matrix(pos_camera):
    PixelCorCenterX = 639.69896228
    PixelCorCenterY = 357.42416711
    fx = 653.39324951
    fy = 634.93518066
    pixel_x = pos_camera[1] * fx / pos_camera[0] + PixelCorCenterX
    pixel_y = pos_camera[2] * fy / pos_camera[0] + PixelCorCenterY
    return [pixel_x, pixel_y]


def calPosAndSendToSimulinkprecise(distance, center_x, center_y, recordPosAndDistance):
    # print("distance0:", distance)
    # f是内参矩阵主对角线的平均 计算xyz，将
    Zc = distance
    # 得到的是估计的位置
    # print("----------------------------")
    # print("distance:", distance)
    # print("----------------------------")
    estimatedroguePos = cal_pos_by_camera_matrix(center_x, center_y, Zc)
    # print("estimatedroguePos: ", estimatedroguePos)
    # print("*****************************\nestimatedroguePos From Camera To Drogue",
    #       estimatedroguePos,
    #       "\n*****************************")
    ddd = 500.0
    if estimatedroguePos[0] > 6:
        dataPyTosimulink = [estimatedroguePos[0] / ddd, estimatedroguePos[1] / ddd, estimatedroguePos[2] / ddd, 1,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0]
    else:
        dataPyTosimulink = [estimatedroguePos[0], estimatedroguePos[1], estimatedroguePos[2], 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0]

    mav.sendCustomData(1, dataPyTosimulink)


def send_to_simulink_vision(estimate_drogue_pos, dis):
    ddd = 500.0
    dis -= estimate_drogue_pos[0] / ddd
    dataPyTosimulink = [estimate_drogue_pos[0] / ddd, (estimate_drogue_pos[1] + 10) / ddd,
                        (estimate_drogue_pos[2] + 10) / ddd, 1,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
    mav.sendCustomData(1, dataPyTosimulink)
    return dis


def send_to_simulink_gps(estimate_drogue_pos, dis):
    ddd = 800.0
    dis -= estimate_drogue_pos[0] / ddd
    dataPyTosimulink = [estimate_drogue_pos[0] / ddd, estimate_drogue_pos[1] / ddd,
                        (estimate_drogue_pos[2] + 5) / ddd, 1,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
    mav.sendCustomData(1, dataPyTosimulink)
    return dis


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

    # flags
    long_gps_flag = True
    short_day_flag = True
    short_night_flag = False
    short_gps_flag = False
    precise_flag = False

    for path, im, im0s, vid_cap, s in dataset:
        path = "1"
        p, im0, frame = path, np.copy(im0s), getattr(dataset, 'frame', 0)
        im0 = im0[0]
        img_depth = dataset.imgs_d[0]

        if long_gps_flag:
            # 远距离阶段
            simulink_data = mav.inSilVect[0].inSILFLoats[:6]
            pos_drogue = simulink_data[:3]
            pos_receiver = simulink_data[3:]
            pos_gps = np.array(pos_drogue) - np.array(pos_receiver)
            estimate_drogue_pos = pos_gps - PosRelativeCameraToReceiver

            dis = send_to_simulink_gps(estimate_drogue_pos, dis)
            # print(dis)
            if dis < d1:
                print("Change to short range stage")
                long_gps_flag = False
        else:
            if not precise_flag:
                # 近距离阶段
                if short_day_flag:
                    # 白天场景
                    im = torch.from_numpy(im).to(device)
                    im = im.half() if half else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                    # Inference save_dir=run/detect/exp
                    visualize = increment_path(save_dir / Path(path).stem,
                                               mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)

                    # NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                               max_det=max_det)

                    for i, det in enumerate(pred):  # per image
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        # add p的路径
                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + (
                            '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        # add 转换
                        im0 = np.array(im0)

                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        # print("len(det)", len(det))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            # img_size是设定的640 im0是取图得到的1280*720
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0[0].shape).round()
                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            # add 给定的序列(包括列表、元组、字符串以及 range(n)区间) reverse只能翻转list
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (
                                        cls, *xywh)  # label format
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                c = int(cls)  # integer class
                                label = None if hide_labels else (
                                    names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                        im0 = annotator.result()

                        im0 = im0[0]

                        # cv2.rectangle(im0, (x10,y10),(x20,y20), (0, 0, 255), 4, 4)
                        # 当标记框飞出边界时，给定框的数值，不至于报错
                        det_arr0 = np.array(det.tolist())
                        if len(det_arr0):
                            if 0 <= det_arr0[0, 0] <= w:
                                x10 = round(det_arr0[0, 0])
                            else:
                                x10 = 0 if det_arr0[0, 0] < 0 else w

                            if 0 <= det_arr0[0, 1] <= h:
                                y10 = round(det_arr0[0, 1])
                            else:
                                y10 = 0 if det_arr0[0, 1] < 0 else h

                            if 0 <= det_arr0[0, 2] <= w:
                                x20 = round(det_arr0[0, 2])
                            else:
                                x20 = 0 if det_arr0[0, 2] < 0 else w

                            if 0 <= det_arr0[0, 3] <= h:
                                y20 = round(det_arr0[0, 3])
                            else:
                                y20 = 0 if det_arr0[0, 3] < 0 else h
                            # print("x10 y20", x10, y10, x20, y20)
                        else:
                            x10, x20, y10, y20 = 0, 1, 0, 1
                        seen += 1
                    center_x = int((x20 - x10) / 2 + x10)
                    center_y = int((y20 - y10) / 2 + y10)
                    estimate_drogue_pos = cal_pos_by_camera_matrix(center_x, center_y, dis)
                    cv2.rectangle(im0, (x10, y10), (x20, y20), (0, 255, 0), 2)
                elif short_night_flag:
                    # 夜晚场景
                    grid_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    grid_hsv = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2HSV)

                    # H、S、V范围一：
                    lower1 = np.array([0, 43, 46])
                    upper1 = np.array([10, 255, 255])
                    mask1 = cv2.inRange(grid_hsv, lower1, upper1)  # mask为二值图像

                    # H、S、V范围二：
                    lower2 = np.array([156, 43, 46])
                    upper2 = np.array([180, 255, 255])
                    mask2 = cv2.inRange(grid_hsv, lower2, upper2)

                    # H、S、V范围三：
                    lower3 = np.array([35, 43, 46])
                    upper3 = np.array([77, 255, 255])
                    mask3 = cv2.inRange(grid_hsv, lower3, upper3)

                    # 将两个二值图像结果 相加
                    mask4 = mask1 + mask2 + mask3

                    lights_index = np.where(mask4 == 255)
                    left_index = (0, 0)
                    right_index = (0, 0)
                    top_index = (0, 0)
                    for i in range(len(lights_index[0])):
                        cur_index = (lights_index[1][i], lights_index[0][i])
                        if cur_index[1] == np.min(lights_index[0]):
                            top_index = cur_index
                        if cur_index[0] == np.min(lights_index[1]):
                            left_index = cur_index
                        elif cur_index[0] == np.max(lights_index[1]):
                            right_index = cur_index

                    center_x = int((left_index[0] + right_index[0]) / 2)
                    center_y = int((left_index[1] + right_index[1]) / 2)
                    estimate_drogue_pos = cal_pos_by_camera_matrix(center_x, center_y, dis)

                    half_w = int(center_x - left_index[0])
                    half_h = int(center_y - top_index[1])

                    cv2.rectangle(im0, (center_x - half_w, center_y - half_h),
                                  (center_x + half_w, center_y + half_h), (0, 255, 0), 2)

                if short_gps_flag:
                    simulink_data = mav.inSilVect[0].inSILFLoats[:6]
                    pos_drogue = simulink_data[:3]
                    pos_receiver = simulink_data[3:]
                    pos_gps = np.array(pos_drogue) - np.array(pos_receiver)
                    estimate_drogue_pos = pos_gps - PosRelativeCameraToReceiver

                if short_gps_flag and (short_day_flag or short_night_flag):
                    center_x_gps, center_y_gps = cal_pixel_by_camera_matrix(estimate_drogue_pos)
                    w_d = (dis - d2) / (d1 - d2)
                    center_x = w_d * center_x_gps + (1 - w_d) * center_x
                    center_y = w_d * center_y_gps + (1 - w_d) * center_y
                    estimate_drogue_pos = cal_pos_by_camera_matrix(center_x, center_y, dis)

                dis = send_to_simulink_vision(estimate_drogue_pos, dis)
                # print(dis)
                if dis < d2:
                    print("Change to drogue detection mode")
                    precise_flag = True

                # print("x_show", center_x)
                # print("y_show", center_y)
            else:
                h, w = img_depth.shape

                img_depth = np.where(img_depth <= 3650, 65535, img_depth)
                min_val = np.min(img_depth)
                depth_coordinates = np.where((img_depth >= min_val) & (img_depth <= min_val + 600))

                # print(len(depth_coordinates[0]))
                if len(depth_coordinates[0]) >= 100:
                    depth_box = [[np.min(depth_coordinates[1]), np.min(depth_coordinates[0])],
                                 [np.max(depth_coordinates[1]), np.max(depth_coordinates[0])]]

                    expand = 10
                    bgr_box = [[max(depth_box[0][0] - expand, 0),
                                max(depth_box[0][1] - expand, 0)],
                               [min(depth_box[1][0] + expand, w - 1),
                                min(depth_box[1][1] + expand, h - 1)]]

                    img_sleeve = im0[bgr_box[0][1]: bgr_box[1][1], bgr_box[0][0]: bgr_box[1][0], :]
                    img_gray = cv2.cvtColor(img_sleeve, cv2.COLOR_BGR2GRAY)
                    img_blur = cv2.medianBlur(img_gray, 5)
                    _, img_binary = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)

                    circles = cv2.HoughCircles(img_binary, cv2.HOUGH_GRADIENT_ALT, 1, 15,
                                               param1=300, param2=0.8, minRadius=0, maxRadius=0)
                    # print(circles)

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

                    recordPosAndDistance.updateMeanStd()
                    recordPosAndDistance.insertNewXYR(center_x, center_y, radius)

                x_show = int(recordPosAndDistance.center_x[-1])
                y_show = int(recordPosAndDistance.center_y[-1])
                r_show = int(recordPosAndDistance.estimate_r[-1])

                cv2.circle(im0, (x_show, y_show), r_show, (0, 255, 0), 2)
                size = 20
                left = int(x_show - 0.5 * size)
                right = int(x_show + 0.5 * size)
                up = int(y_show - 0.5 * size)
                down = int(y_show + 0.5 * size)
                cv2.line(im0, (left, y_show), (right, y_show), (0, 255, 0), 1)
                cv2.line(im0, (x_show, up), (x_show, down), (0, 255, 0), 1)

                # print("x_show", x_show)
                # print("y_show", y_show)
                # 根据rgb图像的中心点坐标和深度图像来计算距离，进而估算锥套位置  把这部分拆除出去，不要影响取图速度
                calPosAndSendToSimulinkprecise(float(min_val / 1000.0), x_show, y_show, recordPosAndDistance)
        cv2.imshow("im0_rgb", im0)
        cv2.imshow('img_depth', img_depth)
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'best.pt',
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
