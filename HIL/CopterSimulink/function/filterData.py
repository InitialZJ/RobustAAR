import numpy as np


class RecordPosAndDistance:
    def __init__(self):
        # 使用前3个frame
        self.frame = 3
        # 判断是否加入数组的3*sigma
        self.N = 3

        self.center_x = [0] * self.frame
        self.center_y = [0] * self.frame
        self.estimate_r = [0] * self.frame
        # 这个以后再优化，现在是通过判断xy来计算distance，
        # 现在每次都要计算。后面可以专门给一个distance，满足条件的话就不用计算
        # self.lidar_distance = [0]

        # 这里给的mean和std少一个，原因是要更新一下才能有第3个数据
        self.mean_center_x = [0] * (self.frame - 1)
        self.mean_center_y = [0] * (self.frame - 1)
        self.mean_estimate_r = [0] * (self.frame - 1)

        self.std_center_x = [0] * (self.frame - 1)
        self.std_center_y = [0] * (self.frame - 1)
        self.std_estimate_r = [0] * (self.frame - 1)

    def updateMeanStd(self):
        #  根据历史记录计算xyr的均值和方差
        self.mean_center_x.append(np.mean(self.center_x[-self.frame:]))
        self.std_center_x.append(np.std(self.center_x[-self.frame:]))
        self.mean_center_y.append(np.mean(self.center_y[-self.frame:]))
        self.std_center_y.append(np.std(self.center_y[-self.frame:]))
        self.mean_estimate_r.append(np.mean(self.estimate_r[-self.frame:]))
        self.std_estimate_r.append(np.std(self.estimate_r[-self.frame:]))

    def insertNewXYR(self, new_x, new_y, new_r):
        # 前3帧直接加入 3+3 就是6，然后第七帧就能开始正常计算了
        if len(self.center_x) <= 2 * self.frame:
            self.center_x.append(new_x)
            self.center_y.append(new_y)
            self.estimate_r.append(new_r)
        else:
            # 根据mean和std判断新增加的数据是否能够加入,如果能加入就直接添加，不能加入就保持现在的数值 用self.xx[-1]代表使用上一次最后的值
            if self.mean_center_x[-1] - self.N * self.std_center_x[-1] <= new_x <= self.mean_center_x[-1] + self.N * self.std_center_x[-1]:
                self.center_x.append(new_x)
            else:
                self.center_x.append(self.center_x[-1])

            if self.mean_center_y[-1] - self.N * self.std_center_y[-1] <= new_y <= self.mean_center_y[-1] + self.N *self.std_center_y[-1]:
                self.center_y.append(new_y)
            else:
                self.center_y.append(self.center_y[-1])

            if self.mean_estimate_r[-1] - self.N * self.std_estimate_r[-1] <= new_r <= self.mean_estimate_r[-1] + self.N * self.std_estimate_r[-1]:
                self.estimate_r.append(new_r)
            else:
                self.estimate_r.append(self.center_x[-1])

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

# new_x = 6
# new_y = 5
# new_r = 5
#
# recordPosAndDistance = RecordPosAndDistance()
# recordPosAndDistance.updateMeanStd()
# recordPosAndDistance.printDataX()
#
#
# recordPosAndDistance.insertNewXYR(new_x, new_y, new_r)
# recordPosAndDistance.updateMeanStd()
# recordPosAndDistance.printDataR()