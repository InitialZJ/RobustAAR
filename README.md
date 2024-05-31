# RobustAAR

<img src="https://gw.alipayobjects.com/zos/antfincdn/R8sN%24GNdh6/language.svg" width="18"> English | [简体中文](./README_zh.md)

This repository contains an all-weather robust visual navigation and localization platform and algorithms. This platform is developed based on the RflySim simulation platform, including tanker, receiver, hose and drogue models, and is developed using Simulink. The docking algorithm is developed using Python, and is divided into three stages: long-range, short-range, and precision docking according to the distance from the tanker to the receiver. In the long-range stage, GPS communication facilitates the approach task in coordination with the tanker. The short-range stage employs a fusion of visual and GPS navigation methods, providing redundant backups to ensure task completion in the event of positioning source loss. Additionally, robust drogue visual localization is achieved during the precision docking stage.

## Getting Started

### Simulation platform

1. Download [RflySim](https://rflysim.com/) simulation platform

2. Double-click `\path\to\PX4PSP\RflySim3D\RflySim3D.exe` to start the simulation and display the default scene

<img src="https://raw.githubusercontent.com/InitialZJ/MarkdownPhotoes/4d34edefbaf7a4a866a8330573be3b534f6ab6a8/res/image-20240530162155493.png?token=ANMBRYQY3NNVDPNF2K62SXLGLBBQ2" alt="image-20240530162155493" style="zoom:50%;" />

3. Open `matlab`, switch to the `\path\to\RobustAAR\AAR_Simulink` directory, open `AirRefueling_Platform.slx`, and click the run button above to generate the `AAR` scene in `RflySim3D`

<img src="https://raw.githubusercontent.com/InitialZJ/MarkdownPhotoes/87d67e0d8fb8c3d2b2b6f610975651ab35a4d776/res/image-20240530164701526.png?token=ANMBRYUESG3BUPVI3OEM77LGLBBVQ" alt="image-20240530164701526" style="zoom:50%;" />

4. The initial view is from the tanker's perspective. Press `B 1` to switch to the receiver's perspective.

<img src="https://raw.githubusercontent.com/InitialZJ/MarkdownPhotoes/42dbcd3c853902ea3ad7e4f21c08a5439295ba33/res/image-20240530164748222.png?token=ANMBRYSC4L63TOYQOCU3JVDGLBBWM" alt="image-20240530171032438" style="zoom:50%;" />

### Algorithms

1. Refer to `\path\to\PX4PSP\PPTs\06.pdf` to configure the `torch` environment

2. Run `\path\to\RobustAAR\Python_Vision_Process\detect-realtime-all.py` to achieve docking

## Experiments

1. **[Daytime all-process experiment](./Python_Vision_Process/all_process/README.md)**

2. **[Nighttime all-process experiment](./Python_Vision_Process/all_process_night/README.md)**

3. **[HIL experiment](./HIL/README.md)**
