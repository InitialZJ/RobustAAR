import csv

import matplotlib.pyplot as plt

import numpy as np

x_ratio = 740.0 / 40
y_ratio = 629.0 / 10

# # for x
# x_list = [0, 31, 58, 128, 167, 212, 282, 331, 392, 463, 517, 574, 622, 664, 688, 708, 740]
# y_list = [652, 622, 583, 492, 440, 385, 310, 261, 223, 180, 144, 109, 69, 36, 19, 9, 0]

# # for y
# x_list = [0, 11, 29, 52, 80, 103, 131, 163, 189, 219, 245, 276, 306, 334, 354, 385, 424, 455, 485, 513, 542, 561, 603, 635, 653, 683, 700, 718, 740]
# y_list = [-46, -33, -21, -37, -70, -89, -92, -86, -78, -66, -55, -52, -56, -65, -68, -62, -50, -47, -48, -40, -33, -29, -37, -45, -47, -39, -29, -17, 0]

# for z
x_list = [0, 36, 66, 112, 156, 187, 217, 248, 278, 307, 338, 353, 384, 414, 460, 491, 535, 582, 612, 658, 740]
y_list = [-46, -39, -39, -37, -33, -31, -26, -22, -17, -11, -5, -3, -7, -8, -9, -12, -11, -9, -11, -6, 0]

x_list = [x / x_ratio for x in x_list]
y_list = [y / y_ratio for y in y_list]

x = np.array(x_list)

y = np.array(y_list)

z1 = np.polyfit(x, y, 9)  # 用3次多项式拟合  可以改为5 次多项式。。。。 返回三次多项式系数

p1 = np.poly1d(z1)

print(p1)  # 在屏幕上打印拟合多项式

yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)

x_save = np.arange(0, 41, 0.1)
y_save = p1(x_save)
csv_file = 'data3.csv'
data = zip(x_save, y_save)
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['t', 'z'])  # 写入 CSV 文件的标题行
    writer.writerows(data)


plot1 = plt.plot(x, y, '*', label='original values')

plot2 = plt.plot(x_save, y_save, 'r', label='polyfit values')

plt.xlabel('xaxis')

plt.ylabel('yaxis')

plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法

plt.title('polyfitting')

plt.show()
