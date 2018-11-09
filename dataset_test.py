import numpy as np
import h5py
import matplotlib.pyplot as plt

f=h5py.File('/media/ashish/C846824B46823A68/Data Sets/NEW_CARLA_Steering_dataset/Data_Set/Ashish3/Training_Data0.h5')
# f = h5py.File('./Final_dataset/Final_data2.h5')

print(list(f.keys()))
t = f['targets']
print(t.dtype)
print(t.shape)
pos = 333
count = 0
# for i in range (t.shape[0]):
#     if t[i, 26] == 0:
#         print(t[i, 0], t[i, 1], t[i, 2], t[i, 26])
#         count = count + 1
    # print(t[i, 26])
    # print(t[i, 0], t[i, 1], t[i, 2], t[i, 26])

print(list(f['Camera'].keys()))
im=f['Camera/RGB_right']
print(im.shape)
plt.imshow(im[pos,:,:,:])
plt.show()
# print(count)
