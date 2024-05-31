import numpy as np
def LocatePos(cameraMatrix, cameraCordinatePos, Zc):
    # 矩阵乘法使用的是@
    return np.linalg.inv(cameraMatrix) @ cameraCordinatePos * Zc

def calPosByCameraMatrix(u, v, Zc):
    cameraMatrix = np.array([[3.2242536777073059e+02, 0., 6.4021106666802694e+02],
                             [0., 3.1992948642842612e+02, 3.6089779926286909e+02],
                             [0., 0., 1.]])
    difCoeffs = np.array([7.4928583195465657e-04, 1.2959153257680841e-02, 1.6026029960792838e-03,
                          2.0035599603467217e-03, 1.5143383348978485e-02])
    cameraCordinatePos = np.array([[u], [v], [1]])
    # 计算出来的结果是[[x],[y],[z]]
    calpos = LocatePos(cameraMatrix, cameraCordinatePos, Zc)
    # 化为一维数据
    calPos = [j for i in calpos for j in i]
    # calPos[-1] = -1 * calPos[-1]

    return [calPos[2], calPos[1], -calPos[0]]
# [8.472931665618985, 0.10864164760923921, 0.13694025407632646]
# [8.472931665618985, 0.10864164760923921, -0.12584712286617009]
dis = 8.475
x=640
y=365
py=360
px=640
f = 320# [8.440410788809992, 2.9102482284549804, 0.7649122277359055]
# # [     7.9242     0.47434      -1.458]
# alpha_z = np.arctan((y - py) / f)
# # 锥套的距离 x方向
# x = dis * np.cos(alpha_z)
# # 锥套的上下偏移 属于z轴方向
# z = dis * np.sin(alpha_z)
# # 锥套的左右偏移，属于y轴方向
# alpha_x = np.arctan((x - px) / f)
# y = dis * np.sin(alpha_x)
# estimateDrougePos = [x, y, z]
# print("estimateDrougePos:",estimateDrougePos)
# print("yc:",yc)
uv = np.sqrt((x - px) * (x - px) + (y - py) * (y - py))
print("uv:",uv)
alpha = np.arctan(uv/f)
Zc = dis*np.cos(alpha)
print("Zc:", Zc)
estimateDrougePos = calPosByCameraMatrix(x, y, Zc)

print("estimateDrougePos From Camera To Drogue", estimateDrougePos)