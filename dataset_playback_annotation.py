import numpy as np
import h5py
import matplotlib.pyplot as plt
import pylab as pl
import time

DIRNAME = './Data_Set/Ashish_Noisy/'
#DIRNAME = './Final_dataset/'
f = h5py.File('/media/ashish/C846824B46823A68/DataSets/NEW_CARLA_Steering_dataset/Data_Set/Ashish/Training_Data11.h5')
t = f['targets']
im = f['Camera/RGB_front']
img = None
print(im.shape[0])

for index in range(t.shape[0]):
    action_abstraction_str = ''
    if img is None:
        img=pl.imshow(im[index, :, :, :])
    else:
        img.set_data(im[index, :, :, :])
        pl.pause(0.001)
        pl.draw()
    # action_abstraction = int(t[index, 25])
    # if action_abstraction & 0b0000000000000010:
    #     action_abstraction_str = action_abstraction_str + 'SpeedUp '
    # if action_abstraction & 0b0000000000000001:
    #     action_abstraction_str = action_abstraction_str + 'DoNothing '
    # if action_abstraction & 0b0000000000000100:
    #     action_abstraction_str = action_abstraction_str + 'SlowDown '
    # if action_abstraction & 0b0000000000001000:
    #     action_abstraction_str = action_abstraction_str + 'Stop '
    # if action_abstraction & 0b0000000000010000:
    #     action_abstraction_str = action_abstraction_str + 'TurnLeft '
    # if action_abstraction & 0b0000000000100000:
    #     action_abstraction_str = action_abstraction_str + 'TurnRight '
    # if action_abstraction & 0b0000000001000000:
    #     action_abstraction_str = action_abstraction_str + 'CutOut '
    # if action_abstraction & 0b0000000010000000:
    #     action_abstraction_str = action_abstraction_str + 'CutIn '
    #
    # visual_abstraction_str = ''
    #
    # visual_abstraction = int(t[index, 26])
    # if visual_abstraction & 0b0000000000000001:
    #     visual_abstraction_str = visual_abstraction_str + 'Stationary_Vehicle '
    # if visual_abstraction & 0b0000000000000010:
    #     visual_abstraction_str = visual_abstraction_str + 'Moving_Vehicle '
    # if visual_abstraction & 0b0000000000000100:
    #     visual_abstraction_str = visual_abstraction_str + 'Stationary_opposite_lane '
    # if visual_abstraction & 0b0000000000001000:
    #     visual_abstraction_str = visual_abstraction_str + 'Moving_Opposite_lane '
    # if visual_abstraction & 0b0000000000010000:
    #     visual_abstraction_str = visual_abstraction_str + 'On_lane '
    # if visual_abstraction & 0b0000000000100000:
    #     visual_abstraction_str = visual_abstraction_str + 'Opposite_lane '
    # if visual_abstraction & 0b0000000001000000:
    #     visual_abstraction_str = visual_abstraction_str + 'On_Sidewalk '
    # if visual_abstraction & 0b0000000010000000:
    #     visual_abstraction_str = visual_abstraction_str + 'Intersection_approaching '
    # if visual_abstraction & 0b0000000100000000:
    #     visual_abstraction_str = visual_abstraction_str + 'Left_turn '
    # if visual_abstraction & 0b0000001000000000:
    #     visual_abstraction_str = visual_abstraction_str + 'Right_turn '
    # if visual_abstraction & 0b0000010000000000:
    #     visual_abstraction_str = visual_abstraction_str + 'Pedestrain_approaching '
    # if visual_abstraction & 0b0000100000000000:
    #     visual_abstraction_str = visual_abstraction_str + 'Pedestrain_departing '
    #
    # print(index)
    # print('Action ', action_abstraction_str)
    # print("Visual ", visual_abstraction_str)

    print('Dis same ', t[index, 34])
    print('Vel same ', t[index, 35])
    print('Dis oppo ', t[index, 36])
    print('Vel oppo ', t[index, 37])
    print('Dis inter ', t[index, 40])
    print('Dis left ', t[index, 41])
    print('Dis right ', t[index, 42])
    print('LDis ped approach ', t[index, 43])
    print('SDis ped approach ', t[index, 44])
    print('LDis ped depart ', t[index, 45])
    print('SDis ped depart ', t[index, 46])

    print('\n')
