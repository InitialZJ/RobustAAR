import csv
import math

import matplotlib.pyplot as plt


def generateXYZ(t):
    x = 6 * math.sin(0.5 * t)
    y = math.sin(2.5 * 0.6 * t + math.pi)
    z = math.sin(2.5 * t)
    return [x - 11, y - 1, z + 1.3]


if __name__ == '__main__':
    t = 0
    t_total = 14
    li_t, li_x, li_y, li_z = [], [], [], []
    while t <= t_total:
        li_t.append(t)
        x, y, z = generateXYZ(t)
        li_x.append(x)
        li_y.append(y)
        li_z.append(z)
        t += 0.2

    csv_file = 'data.csv'
    li_xx = []
    li_yy = []
    li_zz = []

    # 从 CSV 文件中读取数据，并将数据加载到四个列表中
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            li_xx.append(float(row[0]))
            li_yy.append(float(row[1]))
            li_zz.append(float(row[2]))

    plt.plot(li_t, li_x, linewidth=2, color='#024097', label='Real Value')
    plt.plot(li_t, li_y, linewidth=2, color='#024097')
    plt.plot(li_t, li_z, linewidth=2, color='#024097')

    plt.plot(li_t, li_xx, linewidth=2, linestyle='--', color='#078f3c', label='Vision Measurement')
    plt.plot(li_t, li_yy, linewidth=2, linestyle='--', color='#078f3c')
    plt.plot(li_t, li_zz, linewidth=2, linestyle='--', color='#078f3c')

    plt.legend()
    # plt.savefig('fig1.svg', format='svg')
    plt.show()
