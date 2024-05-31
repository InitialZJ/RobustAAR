import numpy as np
# 将多维list转化为1维
pos = [1,2,3]
d = [4]
c = [pos,d,[0]*16]
cc = [j for i in c for j in i]
print(cc)