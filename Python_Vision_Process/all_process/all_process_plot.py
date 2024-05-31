import csv
import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline


def change_list(li_t, lst, li_t_new, dock):
    lst = [-x for x in lst]
    temp = lst[dock]
    lst = [x - temp for x in lst]
    return make_interp_spline(li_t, lst)(li_t_new)


def change_lists(li_t, li_x, li_y, li_z, end, dock):
    li_t_new = np.linspace(li_t[0], li_t[end - 1], 100)
    return li_t_new, change_list(li_t[:end], li_x[:end], li_t_new, dock), change_list(li_t[:end], li_y[:end], li_t_new, dock), change_list(li_t[:end], li_z[:end], li_t_new, dock)


if __name__ == '__main__':
    csv_file = 'data.csv'
    li_t = []
    li_x = []
    li_y = []
    li_z = []

    # 从 CSV 文件中读取数据，并将数据加载到四个列表中
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            li_t.append(float(row[0]))
            li_x.append(float(row[1]))
            li_y.append(float(row[2]))
            li_z.append(float(row[3]))

    li_t, li_x, li_y, li_z = change_lists(li_t, li_x, li_y, li_z, 3500, 2600)
    plt.plot(li_t, li_x, linewidth=2, color='#0072BD', label='x')
    plt.plot(li_t, li_y, linewidth=2, color='#D95319', label='y')
    plt.plot(li_t, li_z, linewidth=2, color='#EDB120', label='z')
    
    plt.legend()
    plt.show()
