import numpy as np
import h5py
import matplotlib.pyplot as plt
import pylab as pl
import time

DIRNAME = './Data_Set/Ashish_Noisy/'
#DIRNAME = './Final_dataset/'
f=h5py.File('/media/ashish/C846824B46823A68/Data Sets/NEW_CARLA_Steering_dataset/Final_dataset/Final_data.h5')
t=f['targets']
# im=f['Camera/RGB_front']
# img=None
# print(im.shape[0])
count = [0]*20

'''
[29927, 47272, 19003, 15024, 10894, 10317, 806, 1585]
'''


for index in range(t.shape[0]):
    command_str = ''
    # if img is None:
    #     img=pl.imshow(im[index, :, :, :])
    # else:
    #     img.set_data(im[index, :, :,:])
    #     pl.pause(0.1)
    #     pl.draw()
    action = int(t[index, 25])
    if action & 0b0000000000000010:
        command_str = command_str + 'SpeedUp '
        count [1] = count [1] + 1
    if action & 0b0000000000000001:
        command_str = command_str + 'DoNothing '
        count[0] = count[0] + 1
    if action & 0b0000000000000100:
        command_str = command_str + 'SlowDown '
        count[2] = count[2] + 1
    if action & 0b0000000000001000:
        command_str = command_str + 'Stop '
        count[3] = count[3] + 1
    if action & 0b0000000000010000:
        command_str = command_str + 'TurnLeft '
        count[4] = count[4] + 1
    if action & 0b0000000000100000:
        command_str = command_str + 'TurnRight '
        count[5] = count[5] + 1
    if action & 0b0000000001000000:
        command_str = command_str + 'CutOut '
        count[6] = count[6] + 1
    if action & 0b0000000010000000:
        command_str = command_str + 'CutIn '
        count[7] = count[7] + 1

    visual = int(t[index, 26])
    if visual & 0b0000000000000001:
        command_str = command_str + 'SpeedUp '
        count [8] = count [8] + 1
    if visual & 0b0000000000000010:
        command_str = command_str + 'DoNothing '
        count[9] = count[9] + 1
    if visual & 0b0000000000000100:
        command_str = command_str + 'SlowDown '
        count[10] = count[10] + 1
    if visual & 0b0000000000001000:
        command_str = command_str + 'Stop '
        count[11] = count[11] + 1
    if visual & 0b0000000000010000:
        command_str = command_str + 'TurnLeft '
        count[12] = count[12] + 1
    if visual & 0b0000000000100000:
        command_str = command_str + 'TurnRight '
        count[13] = count[13] + 1
    if visual & 0b0000000001000000:
        command_str = command_str + 'CutOut '
        count[14] = count[14] + 1
    if visual & 0b0000000010000000:
        command_str = command_str + 'CutIn '
        count[15] = count[15] + 1
    if visual & 0b0000000100000000:
        command_str = command_str + 'SpeedUp '
        count[16] = count [16] + 1
    if visual & 0b0000001000000000:
        command_str = command_str + 'DoNothing '
        count[17] = count[17] + 1
    if visual & 0b0000010000000000:
        command_str = command_str + 'SlowDown '
        count[18] = count[18] + 1
    if visual & 0b0000100000000000:
        command_str = command_str + 'Stop '
        count[19] = count[19] + 1

    print(index)
    # print(command_str)
    print(count)
# for index in range(im.shape[0]-500, im.shape[0]):
#     if img is None:
#         img=pl.imshow(im[index, :, :, :])
#     else:
#         img.set_data(im[index, :, :,:])
#     pl.pause(.001)
#     pl.draw()
#     print(index)