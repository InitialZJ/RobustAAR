import numpy as np
z = 6.5
pitch = 0.1

h = z * np.sin(pitch * np.pi / 180)
print("h", h)