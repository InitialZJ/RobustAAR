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

detectFlag = 1

# 定义服务器地址和端口号
SERVER_IP = '127.0.0.1'
PORT = 20142
# 创建 UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

def GetDataFromTanker():
    global Dis_of_Tanker
    #print("entering GetData...")
    #print("mav.inSilVect length", mav.inSilVect)
    # mav = PX4MavCtrl.PX4MavCtrler(20010)
    # mav.initUE4MsgRec()
    # time.sleep(1)
    # get data from simulink
    # struct PX4SILIntFloat{
    # int checksum;//1234567897
    # int CopterID;
    # int inSILInts[8];
    # float inSILFLoats[20];
    # 这里传输的是位置速度和四元数[ v_x, v_y, v_z, pos_x, pos_y, pos_z, yaw, pitch, row, 11 * 0]  yaw pitch roll(Chi, theta, phi)
    # };
    while True:
        if len(mav.inSilVect) > 0:
            # 这里的Info_of_Tanker就是一个20维的数组
            Info_of_Tanker = mav.inSilVect[0].inSILFLoats
            PosRelativeDrogueToTankerFromSimulink = Info_of_Tanker[:3]
            AngelDrogueFromSimulink = Info_of_Tanker[3:6]
            DeltaReceiverPos = Info_of_Tanker[6:9]
            Dis_of_Tanker = Info_of_Tanker[3] #simulink中加油机位置-x坐标
            # print(Dis_of_Tanker)
            scale = 2
            pointToDrogurCenter = PosOilinToDrogue[0] * scale
            #print("AngelDrogueFromSimulink:",AngelDrogueFromSimulink)
            deltaXDrogue = pointToDrogurCenter * np.cos(AngelDrogueFromSimulink[1] - np.pi)
            deltaZDrogue = pointToDrogurCenter * np.sin(AngelDrogueFromSimulink[1] - np.pi)
            PosRelativePointToDrogue = np.array([-deltaXDrogue, 0, deltaZDrogue])
            TrueRelativeFromTankerToDrogue = np.array(PosRelativeDrogueToTankerFromSimulink) - (
                        np.array(PosRelativeReceiverToTanker) + np.array(PosRelativeCameraToReceiver) + np.array(
                    DeltaReceiverPos)) + PosRelativePointToDrogue
            # print("*************************\nDrogue position To Tanker from simulink From camera To Drogue :{}".format(
            #     TrueRelativeFromTankerToDrogue), "\n***********************")

            if Shared_info.full():
                tmp = Shared_info.get()
                Shared_info.put(Info_of_Tanker)
            # 将获取到的信息加入queue
            else:
                Shared_info.put(Info_of_Tanker)
            # Shared_info.task_done()
            # print("function GetDataFromTanker :Record the info")
        else:
            time.sleep(1)
            print('End')

def ReceiveDockingError():
    global obj_x
    global obj_y
    global obj_z
    global tough_z
    #设置named pipe
    pipe_name = r'\\.\pipe\TestPipe'
    pipe = win32pipe.CreateNamedPipe(
        pipe_name,
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
        1, 65536, 65536,
        0,
        None)
    print("Waiting for client to connect to named pipe...")
    win32pipe.ConnectNamedPipe(pipe, None)
    while True:
        if detectFlag == 2:
            # 读取数据
            result, data = win32file.ReadFile(pipe, 64 * 1024)
            if result == 0:
                # 将接收到的JSON字符串反序列化为字典
                received_data = json.loads(data.decode())
                obj_x = received_data["float1"]
                obj_y = received_data["float2"]
                obj_z = received_data["float3"]
                tough_z = received_data["float4"]
                # print(obj_x, obj_y, obj_z)
    win32file.CloseHandle(pipe)

# 通过像素点计算位置 Zc是垂直距离，激光雷达得到的不是
def LocatePos(cameraMatrix, cameraCordinatePos, Zc):
    # 矩阵乘法使用的是@
    return np.linalg.inv(cameraMatrix) @ cameraCordinatePos * Zc

# 通过位置计算像素点
def LocatePixel(cameraMatrix, relatedToCameraPos, Zc):
    return cameraMatrix @ relatedToCameraPos / Zc

def SimpleRoute():
    global detectFlag

    # Create MAVLink control API instance
    print("entering SimpleRoute...")
    # mavN --> 20100 + (N-1)*2
    mav.InitMavLoop()
    lastTime = time.time()
    startTime = time.time()
    flagTime = time.time()
    timeInterval = 1 / 100.0  # here is 0.0333s (30Hz) timer intervals
    detectFlag = 1
    # 直线轨迹，设置2个航点
    n = 2
    # startpoint = [200, 0, -100]
    # 航点分别是[200 ~],[300 ~],[400 ~]
    startpoint = [250, 119, -200]
    missionPoints = []
    for i in range(n):
        x = startpoint[0] + 100 * (i + 1)
        y = startpoint[1]
        z = startpoint[2]
        missionPoints.append([x, y, z])

    # flags for vehicle 1
    flag = 0

    curPosTankerOld = [0, 0, 0]
    targetPos = [0, 0, 0]
    #设置named pipe


    # 这个数字是根据视觉上对接成功时候，计算出来的相对位置可信
    # d_pr_rc = [7.6, 0.50, -2.44]
    # 摄像头的位置
    # d_cam_rc = [4.7, 0.54, -1.45]
    # d_pr_cam = [d_pr_rc[0] - d_cam_rc[0], d_pr_rc[1] - d_cam_rc[1], d_pr_rc[2] - d_cam_rc[2]]

    spd = 10

    # Start a endless loop with 30Hz, timeInterval=1/30.0
    while True:
        lastTime = lastTime + timeInterval
        sleepTime = lastTime - time.time()

        if sleepTime > 0:
            time.sleep(sleepTime)  # sleep until the desired clock 在这个时间段内，说明还没有到达下一个0.0333s，所以睡眠
            continue
        else:
            lastTime = time.time()
        # The following code will be executed 30Hz (0.0333s)

        # Local Pos (related to takeoff position)
        curPos = mav.uavPosNED

        # True simulated position
        truePos = mav.truePosNED

        if Shared_info.full():
            curPosTanker = Shared_info.get()[3:6]
            curPosTankerOld = curPosTanker
        else:
            curPosTanker = curPosTankerOld
        # print(curPosTanker)

        # Create the mission to vehicle 1
        if time.time() - startTime > 5 and flag == 0:
            # The following code will be executed at 5s
            print("5s, Arm the drone")
            flag = 1
            mav.SendMavArm(True)  # Arm the drone
            print("Arm the drone!")

            # 发送命令让飞机起飞，输入的五项分别是，最小俯仰角（单位rad），偏航角（单位rad），期望位置（单位m）X，Y，Z（相对于解锁位置）
            # 如果要发送绝对的GPS坐标作为起飞目标点，请用sendMavTakeOffGPS命令，最后三位分别是经度、维度、和高度，建议先从uavPosGPSHome向量中提取解锁GPS坐标，在此基础上用绝对坐标
            # 刚开始生成的飞机的位置 [-250, -119, 0] 但是curPos = mav.uavPosNED获取的位置是从[0 0 0]开始计算的
            targetPos = [100, 0, -100]
            # 只需要传入起飞位置的xyz即可，在这个指令中，默认pitch=15，yaw=0
            # mav.SendVelNEDNoYaw(10, 0, 0)
            mav.sendMavTakeOff(targetPos[0], targetPos[1], targetPos[2])
            time.sleep(0.5)
            print("开始起飞")
            mav.SendMavArm(True)  # Arm the drone

        elif flag == 1:
            # 从CopterSim里面接受信息
            # print("curPos=", curPos)
            dis = math.sqrt((curPos[0] - targetPos[0]) ** 2 + (curPos[1] - targetPos[1]) ** 2)
            # print("dis=", dis)
            if dis < 50:
                print("到达起飞位置\n")
                flag = 2
                mav.SendVelNEDNoYaw(10, 0, 0)
                mav.initOffboard()
                print("开始进入Offboard模式")
                print("开始进入姿态控制模式")
                mav.SendAttPX4([0, 0, -15, 0], 0.5, 0, 0)

        elif flag == 2:
            mav.SendAttPX4([0, 0, -15, 0], 0.5, 0, 0)
            if time.time() - flagTime > 5:
                print("结束姿态控制模式\n")
                flag = 3
                flagTime = time.time()
                flagI = 0
                flagII = 0
                delta_x_last = 0
        elif flag == 3:
            if time.time() - flagTime > 0.1:
                delta_x = mav.uavPosNED[1]
                print("开始进入速度高度偏航控制模式")
                print(Dis_of_Tanker - mav.uavPosNED[0])
                if (Dis_of_Tanker - mav.uavPosNED[0]) < -190:
                    flagII = 1
                if (Dis_of_Tanker - mav.uavPosNED[0]) < -220:
                    flagII = 2
                if (Dis_of_Tanker - mav.uavPosNED[0]) < -215:
                    detectFlag = 2
                # 发送标志
                sent = sock.sendto(str(detectFlag).encode(), (SERVER_IP, PORT))

                if flagII == 0:
                    v = 20
                elif flagII == 1:
                    v = 14.7
                elif flagII == 2:
                    v = 14.7
                    flag = 4
                # print(delta_x - delta_x_last)
                yaw = (delta_x - 0) / 35 + (delta_x - delta_x_last) / 40
                height = (mav.uavPosNED[2] + 90.6) / 5
                mav.SendVelYawAlt(v, -yaw, -90.6 - height)
                delta_x_last = delta_x
                flagTime = time.time()
                H = -91.2
                px = 639.69896228 + 0
                py = 357.42416711 + 0
                flagIII = 0
                const_z = 0
                const_y = 0
                const_x = 0
                const_zz = 0
                ex_m = []
                ey_m = []
                ez_m = []
                px_m = []
                py_m = []
                pz_m = []
                ex_last = 0
                ey_last = 0

        elif flag == 4:
            if time.time() - flagTime > 0.01:
                delta_z = mav.uavPosNED[0]
                delta_x = mav.uavPosNED[1]
                delta_y = mav.uavPosNED[2]
                print("开始进入图像伺服控制模式")
                print(obj_x, obj_y, obj_z, tough_z)
                # print("图像误差: %f, %f, %f" % ((obj_x - px), (obj_y - py), obj_z))
                if (obj_y - py) >= -300 and obj_z <= 13:
                    flagIII += 1

                if flagIII == 0:
                    # print("x轴误差：%f" % (mav.uavPosNED[1]))
                    # print(delta_x - delta_x_last)
                    yaw = (delta_x - 0) / 15 + (delta_x - delta_x_last) / 15
                    height = (mav.uavPosNED[2] + 91.2) / 10
                    mav.SendVelYawAlt(14.6, -yaw, -91.2 - height)
                    delta_x_last = delta_x
                elif flagIII >= 1:
                    if (obj_y - py) <= -300 or (obj_x - px) <= -600:
                        obj_yy = const_y
                        obj_xx = const_x
                    else:
                        obj_yy = obj_y
                        obj_xx = obj_x
                        const_y = obj_y
                        const_x = obj_x
                    obj_zz = obj_z - 2.67
                    tough_zz = tough_z - 1.54
                    if obj_zz < 0:
                        obj_zz = const_z
                    else:
                        const_z = obj_zz
                    print("图像误差滤波后: %f, %f, %f" % ((obj_xx - px), (obj_yy - py), obj_zz))
                    if obj_zz >= 0.03 or (obj_xx - px) >= 30 or (obj_yy - py) >= 30:
                        ex_m.append((obj_xx - px) * ((obj_zz + 0.1) / 5))
                        ey_m.append((obj_yy - py) * ((obj_zz + 0.1) / 5))
                        ez_m.append(obj_zz - 0)
                        px_m.append(delta_x)
                        py_m.append(-delta_y - 92.4)
                        pz_m.append(Dis_of_Tanker - delta_z + 236)
                        scio.savemat('error_data.mat', mdict={'ex': ex_m, 'ey': ey_m, 'ez': ez_m, 'px': px_m, 'py': py_m, 'pz': pz_m})
                    p = mav.trueAngRate[0] * 0.002
                    q = mav.trueAngRate[1] * 0.002
                    r = mav.trueAngRate[2] * 0.002
                    ex = (obj_xx - px) / 640
                    ey = (obj_yy - py) / 360
                    ez = obj_zz
                    if flagIII == 1:
                        ex_last = ex
                        ey_last = ey
                    d_pr_rc = [6.5, 0, -2.1]
                    Vgt = 14.48

                    k1 = 0.002
                    k2 = 0.002
                    kx_p = 3.62
                    kx_d = 0.12
                    ky_p = 0.1
                    ky_d = 0.1
                    kz = 0.03

                    if abs(obj_xx - px) <= 30 and abs(obj_yy - py) <= 30 and obj_zz <= 0.4:
                        Va = 15
                        Chi = -mav.uavPosNED[1] / 50
                        H = -92.2
                        flag = 5
                        print('对接成功!')
                    else:
                        Vgx = kz*(ez)-k1*abs(ex)-k2*abs(ey)+Vgt-d_pr_rc[2]*q+d_pr_rc[1]*r
                        Vgy = kx_p*(ex)+kx_d*(ex-ex_last)+ez*ex*ey*q-ez*(1+ex**2)*r+ez*ey*p-d_pr_rc[0]*r+d_pr_rc[2]*p
                        Vgz = ky_p*(ey)+ky_d*(ey-ey_last)-ez*ex*ey*r+ez*(1+ey**2)*q-ez*ey*p-d_pr_rc[1]*p+d_pr_rc[0]*q
                        Va = math.sqrt(Vgx**2 + Vgy**2)
                        Chi = math.asin(Vgy/Va)
                        H = Vgz + H
                        # Va = math.sqrt((kz * (ez) - k1 * abs(ex) - k2 * abs(ey) + Vgt) ** 2 + (kx_p * (ex) + kx_d * (ex - ex_last)) ** 2)
                        # Chi = math.asin((kx_p * (ex) + kx_d * (ex - ex_last)) / Va)
                        # H = ky_p * (ey) + ky_d * (ey - ey_last) + H

                    mav.SendVelYawAlt(Va, Chi, H)
                    print(Va, Chi, H, ex - ex_last, ey - ey_last)
                    ex_last = ex
                    ey_last = ey

                flagTime = time.time()
        elif flag == 5:
            yaw = mav.uavPosNED[1] / 50
            height = (mav.uavPosNED[2] + 92.2) / 10
            mav.SendVelYawAlt(15, -yaw, -92.2 - height)


def run_threads():
    getDataFromTanker = td.Thread(target=GetDataFromTanker, args=())
    getDataFromTanker.start()
def receive_dockingerror():
    receivedockingerror = td.Thread(target=ReceiveDockingError, args=())
    receivedockingerror.start()
if __name__ == "__main__":
    run_threads()
    receive_dockingerror()
    SimpleRoute()

