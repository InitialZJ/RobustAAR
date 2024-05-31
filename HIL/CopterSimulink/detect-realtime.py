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
import copy
import os
import sys
from pathlib import Path

# import cv2.cv2 as cv2
import cv2
import matplotlib.pyplot
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
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import PX4MavCtrlV4 as PX4MavCtrl
import threading as td
import math
import time
from queue import Queue
import scipy.io as scio
import socket
import json
import win32pipe, win32file
from multiprocessing import Pipe, Process

# 1为关闭/2为开启(默认关闭）
detectFlag = 1

# 定义服务器地址和端口号
SERVER_IP = '127.0.0.1'
PORT = 20142
# 创建 UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 将Socket绑定到指定的 IP 地址和端口上
sock.bind((SERVER_IP, PORT))

# 准备向simulink中传输数据
# mav = PX4MavCtrl.PX4MavCtrler(20100)
# mav.initUE4MsgRec()
# def InitialPos():
# simulink中设定
PosWorldTanker = np.array([260, 8, -1320])
PosCenterToModel = np.array([0, 0, 0])
# PosRelativeReceiverToTanker = [-40, -1, 14] + PosCenterToModel
PosRelativeReceiverToTanker = np.array([-40, -1, 14])
PosWorldReceiver = PosWorldTanker + PosRelativeReceiverToTanker
# json文件中设定
PosRelativeCameraToReceiver = np.array([6.5, 0, -2.1])
# 实际上下面两个参数并未使用，在Rflysim3D中需要校正
PosRelativeProbeToReceiver = np.array([7.3712, 0.5497, -1.4418])
PosRelativeNoseToReceiver = np.array([9.6940, 0, 0])
# 锥套相关参数
PosOilinToDrogue = np.array([0.347, 0, 0])
PosUmbrelaaToDrogue = np.array([1.3210, 0, 0])
# 这里有过放缩 是之前数据的2倍
PosRelativeDrogueToTanker = np.array([-27.08, -0.142, 9.7862])


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
        if zc <= 2.67 or zc >= 20:
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
            if self.mean_center_x[-1] * (1-rate) <= new_x <= self.mean_center_x[-1] * (1+rate) \
                    and self.mean_center_y[-1] * (1-rate) <= new_y <= self.mean_center_y[-1] * (1+rate) \
                    and self.mean_estimate_r[-1] * (1-rate) <= new_r <= self.mean_estimate_r[-1] * (1+rate):
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
# 暂时没用
Shared_info = Queue(maxsize=1)  # 设置 Queue的最大数 拿到的是受油机的位置信息


# 通过像素点计算位置 Zc是垂直距离，激光雷达得到的不是
def LocatePos(cameraMatrix, cameraCordinatePos, Zc):
    # 矩阵乘法使用的是@
    return np.linalg.inv(cameraMatrix) @ cameraCordinatePos * Zc


# 通过位置计算像素点
def LocatePixel(cameraMatrix, relatedToCameraPos, Zc):
    return cameraMatrix @ relatedToCameraPos / Zc


# Zc是深度相机测量出来的距离，是直线距离
def calPosByCameraMatrix(u, v, Zc):
    cameraMatrix = np.array([[653.39324951, 0., 639.69896228],
                             [0., 634.93518066, 357.42416711],
                             [0., 0., 1.]])
    difCoeffs = np.array([7.4928583195465657e-04, 1.2959153257680841e-02, 1.6026029960792838e-03,
                          2.0035599603467217e-03, 1.5143383348978485e-02])
    PixelCorCenterX = 639.69896228
    PixelCorCenterY = 357.42416711
    fx = 653.39324951
    fy = 634.93518066
    deltaX = Zc * (u - PixelCorCenterX) / fx
    deltaY = Zc * (v - PixelCorCenterY) / fy
    # 返回值的坐标系是世界坐标系，上面计算出来的是相机坐标系
    return [Zc, deltaX, deltaY]


def calPosAndSendToSimulink(img_depth, center_x, center_y, recordPosAndDistance):
    global obj_x
    global obj_y
    global obj_z
    global tough_z
    # 根据识别出来的锥套来确定距离
    distance = float(img_depth[center_y, center_x] / 1000.0)
    # f是内参矩阵主对角线的平均 计算xyz，将
    f = 644
    px = 639.69896228
    py = 357.42416711
    uv = np.sqrt((center_x - px) * (center_x - px) + (center_y - py) * (center_y - py))
    alpha = np.arctan(uv / f)
    # Zc = distance
    Zc = distance * np.cos(alpha)
    # 得到的是估计的位置
    # print("----------------------------")
    # print("distance:", distance)
    # print("----------------------------")
    # 对于Zc的滤波要在这里和xy点一起做
    Zc = recordPosAndDistance.insertZc(Zc)
    estimateDrougePos = calPosByCameraMatrix(center_x, center_y, Zc)
    # print("estimateDrougePos: ", estimateDrougePos)
    # print("*****************************\nestimateDrougePos From Camera To Drogue", estimateDrougePos,
    #       "\n*****************************")
    # 直接从simulink中获得锥套的真实数据
    dataPyTosimulink = [estimateDrougePos[0], estimateDrougePos[1], estimateDrougePos[2], distance, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
    obj_x = float(center_x)
    obj_y = float(center_y)
    obj_z = float(estimateDrougePos[0])
    tough_z = float(distance)
    print(obj_x,obj_y,obj_z,tough_z)
    # mav.sendCustomData(1, dataPyTosimulink)


@torch.no_grad()
# weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
def run(weights=ROOT / 'best.pt',  # model.pt path(s)
        # source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        # source=ROOT / 'https://www.bilibili.com/video/BV1nv4y1K7df?spm_id_from=333.851.b_7265636f6d6d656e64.1',  # file/dir/URL/glob, 0 for webcam
        source='',
        imgsz=640,  # inference size (pixels)
        conf_thres=0.20,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        # max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
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
    # add
    global detectFlag
    save_txt = True
    hide_labels = False

    # named pipe 传输数据
    pipe_name = r'\\.\pipe\TestPipe'
    fileHandle = win32file.CreateFile(
        pipe_name,
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        0, None,
        win32file.OPEN_EXISTING,
        0, None)

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run save-dir=run/detect/exp
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 创建一个存放视频的writer
    record_fps, record_w, record_h = 30, 1280, 720
    # record_writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, (record_w, record_h))
    record_writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*'XVID'), record_fps, (record_w, record_h))

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
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:

        if detectFlag == 1:
            data, client_address = sock.recvfrom(1024)
            detectFlag = int(data.decode())
            print("Received message from {}: {}".format(client_address[0], detectFlag))
            continue
        # 保存此次对接的视频，1.avi
        path = "1"
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference save_dir=run/detect/exp
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()

        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        # print("pred:", pred)
        dt[2] += time_sync() - t3


        for i, det in enumerate(pred):  # per image
            # i是序号 det是对应的值 det这里是(1,6)
            # print("i", i)
            # print("det", det.shape)
            det_arr0 = np.array(det.tolist())
            # print("det_arr0", det_arr0.shape)
            # [[x1,y1,x2,y2]] 在图像上 x是横坐标 y是纵坐标
            w = 1280
            h = 720
            # w = 640
            # h = 480
            # 当标记框飞出边界时，给定框的数值，不至于报错
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

            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            # add p的路径
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
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
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # add 给定的序列(包括列表、元组、字符串以及 range(n)区间) reverse只能翻转list
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or save_crop or view_img:  # Add bbox to image
                    #     c = int(cls)  # integer class
                    #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #     annotator.box_label(xyxy, label, color=colors(c, True))
                    #     if save_crop:
                    #         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    # print("*xyxy",*xyxy)
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

            img_depth = dataset.imgs_d[0]
            cv2.imshow("img_depth", img_depth)

            # Save results (image with detections)

            im0 = im0[0]

            # cv2.rectangle(im0, (x10,y10),(x20,y20), (0, 0, 255), 4, 4)
            center_x = int((x20 - x10) / 2 + x10)
            center_y = int((y20 - y10) / 2 + y10)
            radius = max(int((x20 - x10) / 2), int((y20 - y10) / 2))

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
            cv2.imshow("im0_rgb", im0)
            # cv2.circle(im0_mask, (x_show, y_show), r_show, (0, 255, 0), 2)
            # cv2.imshow("im0_rgb", im0_mask)


            record_writer.write(im0)

            # print("x_show", x_show)
            # print("y_show", y_show)
            # 根据rgb图像的中心点坐标和深度图像来计算距离，进而估算锥套位置  把这部分拆除出去，不要影响取图速度
            calPosAndSendToSimulink(img_depth, x_show, y_show, recordPosAndDistance)

            # 准备发送的float变量
            data_to_send = {
                "float1": obj_x,
                "float2": obj_y,
                "float3": obj_z,
                "float4": tough_z
            }

            json_data = json.dumps(data_to_send)

            # 发送数据
            win32file.WriteFile(fileHandle, json_data.encode())

    win32file.CloseHandle(fileHandle)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default='', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[720, 1280],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 保存我从UE4中取到的视频的地方
    parser.add_argument('--record', default=r"E:\LGY\CV\Python\1-PX4UavCtrlAPI\static\runs\record\1.avi",
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