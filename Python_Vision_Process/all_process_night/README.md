# Nighttime all-process experiment

<img src="https://gw.alipayobjects.com/zos/antfincdn/R8sN%24GNdh6/language.svg" width="18"> English | [简体中文](./README_zh.md)

**Note**: This program requires UE5 support. If necessary, please [obtain](https://rflysim.com/download.html) the advanced version platform

1. Open `\path\to\PX4PSP\RflySimUE5\RflySim3D.exe` and manually switch the map to `Desert`
2. Modify the `ScaleXYZ` value of the drogue module in `AirRefueling_Platform.slx` to `[1,1,1]` and run

<img src="C:/Users/25434/AppData/Roaming/Typora/typora-user-images/image-20240530203324293.png" alt="image-20240530203324293" style="zoom:67%;" />

3. Copy `all_process_night_exp.py` and `all_process_night_plot.py` to the parent directory
4. Run `all_process_night_exp.py` to generate `data.csv`
5. Run `all_process_night_plot.py` to draw the curve graph