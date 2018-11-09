import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob

DIRPATH = '/media/ashish/C846824B46823A68/DataSets/NEW_CARLA_Steering_dataset/Data_Set/'
WINDOW_WIDTH = 320
WINDOW_HEIGHT = 180
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

final_f = h5py.File("/media/ashish/C846824B46823A68/DataSets/NEW_CARLA_Steering_dataset/Final_dataset/Final_data3.h5", "w")

dset_target = final_f.create_dataset('targets', (0, 47), maxshape=(None, 50), dtype=np.float128)
dset_im_front = final_f.create_dataset('Camera/RGB_front', (0, WINDOW_HEIGHT, WINDOW_WIDTH, 3),
                                       maxshape=(None, WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

# dset_im_left = final_f.create_dataset('Camera/RGB_left', (0, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3),
#                                       maxshape=(None, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3), dtype=np.uint8)
# dset_im_right = final_f.create_dataset('Camera/RGB_right', (0, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3),
#                                        maxshape=(None, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3), dtype=np.uint8)
# dset_im = [dset_im_front, dset_im_left, dset_im_right]

# truncated = [[65,1436], [50,800], [55,6900], [40, 1300], [45, 2450], [50, 1250], [50, 3550],  # Anay
#              [40, 3100], [35, 2260], [35, 1800], [45, 760], [95, 980], [35, 1550], [35, 2574], [35, 3230], [35, 3350],
#              [35, 2500],  # Ashish
#              [45, 902], [40, 2168], [55, 2468], [50, 1639], [40, 276], [40, 397], [65, 518], [40, 189], [30, 132],
#               [35, 2162], [45, 3490],
#              [40, 1400], [40, 815], [40, 620], [90, 552], [60, 817], [60, 233], [40, 1077], [45, 1608], #Ashish Noisy
#              [40, 1300], [30, 175], [40, 1370], [45, 4000], [50, 7300], [50, 3320], # Fenil
#              [75, 1900], [60, 4040], [60, 4100], [50, 1650], [70, 650], # Pranesh
#              [55, 350], [55, 1650], [125, 550], [70, 1300], [65, 3800], [85, 1000], [45, 2700]] # Ramesh
# count = 0

# for i in sorted(glob.glob(DIRPATH + '*')):
#     for j in sorted(glob.glob(i + '/*')):
#         print("Processing file:", j)
#         f = h5py.File(j, "r")
#         start = truncated[count][0]
#         stop = truncated[count][1]
#         previous_size = dset_target.shape[0]
#
#         dset_target.resize(dset_target.shape[0] + stop - start, axis=0)
#         dset_target[previous_size:, :] = f['targets'][start:stop]
#         dset_im_left.resize(dset_im_left.shape[0] + stop - start, axis=0)
#         dset_im_left[previous_size:, :, :, :] = f['Camera/RGB_right'][start:stop]
#         dset_im_right.resize(dset_im_right.shape[0] + stop - start, axis=0)
#         dset_im_right[previous_size:, :, :, :] = f['Camera/RGB_right'][start:stop]
#         dset_im_front.resize(dset_im_front.shape[0] + stop - start, axis=0)
#         dset_im_front[previous_size:, :, :, :] = f['Camera/RGB_front'][start:stop]
#
#         count += 1

for i in sorted(glob.glob(DIRPATH + '*')):
    for j in sorted(glob.glob(i + '/*.h5')):
        print("Processing file:", j)
        f = h5py.File(j, "r")
        start = 70
        stop = f['targets'].shape[0] - 80
        previous_size = dset_target.shape[0]

        dset_target.resize(dset_target.shape[0] + stop - start, axis=0)
        dset_target[previous_size:, :] = f['targets'][start:stop]

        # dset_im_left.resize(dset_im_left.shape[0] + stop - start, axis=0)
        # dset_im_left[previous_size:, :, :, :] = f['Camera/RGB_right'][start:stop]
        # dset_im_right.resize(dset_im_right.shape[0] + stop - start, axis=0)
        # dset_im_right[previous_size:, :, :, :] = f['Camera/RGB_right'][start:stop]

        dset_im_front.resize(dset_im_front.shape[0] + stop - start, axis=0)
        dset_im_front[previous_size:, :, :, :] = f['Camera/RGB_front'][start:stop]

        # count += 1
