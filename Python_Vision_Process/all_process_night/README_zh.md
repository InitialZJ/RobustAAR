# 夜间全过程实验

<img src="https://gw.alipayobjects.com/zos/antfincdn/R8sN%24GNdh6/language.svg" width="18"> [English](./README.md) | 简体中文

**注意**：本程序需要 UE5 支持，如有需要请[获取](https://rflysim.com/download.html)高级版平台

1. 打开 `\path\to\PX4PSP\RflySimUE5\RflySim3D.exe`，手动切换地图为 `Desert`
2. 修改 `AirRefueling_Platform.slx` 中锥套模块的 `ScaleXYZ` 值为 `[1,1,1]`，最后运行

<img src="C:/Users/25434/AppData/Roaming/Typora/typora-user-images/image-20240530203324293.png" alt="image-20240530203324293" style="zoom:67%;" />

3. 将 `all_process_night_exp.py` 和 `all_process_night_plot.py` 拷贝到上一级目录
4. 运行 `all_process_night_exp.py` 生成 `data.csv`
5. 运行 `all_process_night_plot.py` 绘制曲线图