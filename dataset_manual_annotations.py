import numpy as np
import h5py
import matplotlib.pyplot as plt
import pylab as pl
import time
import glob

DIRNAME = '/media/ashish/C846824B46823A68/Data Sets/CARLA_Steering_dataset/Data_Set/'
print(DIRNAME)
for i in sorted(glob.glob(DIRNAME + '*')):
    for j in sorted(glob.glob(i + '/*')):
        print("Processing file:", j)
        f = h5py.File(j, "r+")
        t=f['targets']
        im=f['Camera/RGB_front']
        img=None
        print(f['targets'].shape[0])
        annotation = 0b0
        for index in range(2450, f['targets'].shape[0]):
            if img is None:
                 img = pl.imshow(im[index, :, :, :])
            else:
                 img.set_data(im[index, :, :, :])
            pl.pause(0.01)
            pl.draw()
            print("Steer {}, Gas {}, Brake {}".format(f['targets'][index, 0], f['targets'][index, 1],
                                                       f['targets'][index, 2]))
            #  char_in = input('Frame {} Enter annotation 0 do nothing | 1 speed up | 2 slow down | 3 stop | 4 turn left | '
            #                  '5 turn right | 6 cut out | 7 cut in: '.format(index))
            char_in = input('Frame {}'.format(index))
            if char_in == '':

                f['targets'][index, 26] = annotation
                print('Annotation:', f['targets'][index, 26])

            else:
                annotation = 0b0000000000000000
                while(char_in != ''):
                    if char_in == '0':
                        annotation = annotation | 0b0000000000000001
                        print('Do nothing')
                    elif char_in == '1':
                        annotation = annotation | 0b0000000000000010
                        print('Speed Up')
                    elif char_in == '2':
                        annotation = annotation | 0b0000000000000100
                        print('Slow Down')
                    elif char_in == '3':
                        annotation = annotation | 0b0000000000001000
                        print('Stop')
                    elif char_in == '4':
                        annotation = annotation | 0b0000000000010000
                        print('Turn Left')
                    elif char_in == '5':
                        annotation = annotation | 0b0000000000100000
                        print('Turn Right')
                    elif char_in == '6':
                        annotation = annotation | 0b0000000001000000
                        print('Cut out')
                    elif char_in == '7':
                        annotation = annotation | 0b0000000010000000
                        print('Cut in')
                    elif char_in == 'u':
                        annotation = 0b0
                        print('undo')
                    # char_in = input('Enter annotation 0 do nothing | 1 speed up | 2 slow down | 3 stop | 4 turn left | '
                    #                 '5 turn right | 6 cut out | 7 cut in: ')
                    char_in = input('Frame {}'.format(index))
                f['targets'][index, 26] = annotation
                # print('Annotation:', f['targets'][index, 26])