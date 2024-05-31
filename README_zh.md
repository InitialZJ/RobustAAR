# RobustAAR

<img src="https://gw.alipayobjects.com/zos/antfincdn/R8sN%24GNdh6/language.svg" width="18"> [English](./README.md) | 简体中文

本仓库包含了一套 AAR 仿真平台，以及全天候鲁棒对接算法。此平台基于 RflySim 仿真平台开发，包括加油机、受油机、软管和锥套模型，使用 Simulink 进行开发。对接算法使用 Python 开发，据加油机到受油机的距离划分为远距离、近距离与精确对接三个阶段，在远距离阶段使用 GPS 与加油机进行沟通完成靠近任务；在近距离阶段采用视觉与 GPS 融合的方法进行导航，两者形成冗余备份，任意定位源丢失都能继续完成任务；在精确对接阶段实现了鲁棒的锥套视觉定位。

## 开始使用

### 仿真平台

1. 下载 [RflySim](https://rflysim.com/) 仿真平台

2. 双击 `\path\to\PX4PSP\RflySim3D\RflySim3D.exe` 启动仿真，显示默认场景

<img src="https://raw.githubusercontent.com/InitialZJ/MarkdownPhotoes/4d34edefbaf7a4a866a8330573be3b534f6ab6a8/res/image-20240530162155493.png?token=ANMBRYQY3NNVDPNF2K62SXLGLBBQ2" alt="image-20240530162155493" style="zoom:50%;" />

3. 打开 `matlab`，切换到 `\path\to\RobustAAR\AAR_Simulink` 目录，打开 `AirRefueling_Platform.slx`，点击上方的运行按钮，将在 `RflySim3D` 中生成 `AAR` 场景

<img src="https://raw.githubusercontent.com/InitialZJ/MarkdownPhotoes/87d67e0d8fb8c3d2b2b6f610975651ab35a4d776/res/image-20240530164701526.png?token=ANMBRYUESG3BUPVI3OEM77LGLBBVQ" alt="image-20240530164701526" style="zoom:50%;" />

4. 初始时刻是加油机视角，按下 `B 1` 切换到受油机视角

<img src="https://raw.githubusercontent.com/InitialZJ/MarkdownPhotoes/42dbcd3c853902ea3ad7e4f21c08a5439295ba33/res/image-20240530164748222.png?token=ANMBRYSC4L63TOYQOCU3JVDGLBBWM" alt="image-20240530171032438" style="zoom:50%;" />

### 算法

1. 参考 `\path\to\PX4PSP\PPTs\RflySim高级版_第06讲_视觉控制算法开发.pdf`，配置 `torch` 环境
2. 运行 `\path\to\RobustAAR\Python_Vision_Process\detect-realtime-all.py`，实现对接

## 实验

1. **[白天全过程实验](./Python_Vision_Process/all_process/README_zh.md)**

2. **[夜间全过程实验](./Python_Vision_Process/all_process_night/README_zh.md)**

3. **[硬件在环实验](./HIL/README_zh.md)**