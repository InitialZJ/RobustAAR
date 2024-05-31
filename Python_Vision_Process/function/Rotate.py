import numpy as np


roll=0
pitch=0.5*np.pi
yaw =0


def calRotateX(roll):
    Rotata_x = roll
    return np.array([[1, 0, 0],
                     [0, np.cos(Rotata_x), -np.sin(Rotata_x)],
                     [0, np.sin(Rotata_x), np.cos(Rotata_x)]])

def calRotateY(pitch):
    Rotata_y = pitch
    return np.array([[np.cos(Rotata_y), 0, np.sin(Rotata_y)],
                     [0, 1, 0],
                     [-np.sin(Rotata_y), 0, np.cos(Rotata_y)]])
def calRotateZ(yaw):
    Rotata_z = yaw
    return np.array([[np.cos(Rotata_z), -np.sin(Rotata_z), 0],
                     [np.sin(Rotata_z), np.cos(Rotata_z), 0],
                     [0, 0, 1]])

def eulerToRotateMatrix(roll, pitch, yaw, flag = "xyz"):
    if flag == "xyz":
        return calRotateZ(roll)@calRotateY(pitch)@calRotateX(yaw)

matrix = eulerToRotateMatrix(roll, pitch, yaw)
print(matrix)



