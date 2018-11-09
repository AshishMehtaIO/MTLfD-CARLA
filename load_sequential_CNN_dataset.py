import numpy as np
import h5py
import random
from imgaug import augmenters as iaa
import copy
import matplotlib.pyplot as plt

f = h5py.File('/media/ashish/C846824B46823A68/DataSets/NEW_CARLA_Steering_dataset/Final_dataset/Final_data3.h5', 'r')


class Dataset_generators():

    def __init__(self, single_image=True, augmentation=True, preload_data=True, only_lane_follow=False,
                 sequential_data=False, validation_set_size=10000, seq_len=6):
        self._single_image = single_image
        self._augmentation = augmentation
        self._preload_data = preload_data
        self._only_lane_follow = only_lane_follow
        self._sequential_data = sequential_data
        self._validation_set_size = validation_set_size
        self._seq_len = seq_len

    @staticmethod
    def partition(alist, indices):
        return [alist[i:j] for i, j in zip([0] + indices, indices + [None])]

    def apply_augmentation(self, images, batchSize):
        st = lambda aug: iaa.Sometimes(0.4, aug)

        oc = lambda aug: iaa.Sometimes(0.3, aug)
        rl = lambda aug: iaa.Sometimes(0.09, aug)
        augment = iaa.Sequential([
            rl(iaa.GaussianBlur((0, 1.5))),  # blur images with a sigma between 0 and 1.5
            rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
            oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),  # randomly remove up to X% of the pixels
            oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)),
            # randomly remove up to X% of the pixels
            oc(iaa.Add((-40, 40), per_channel=0.5)),  # change brightness of images (by -X to Y of original value)
            st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),  # change brightness of images (X-Y% of original value)
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),  # improve or worsen the contrast
            rl(iaa.Grayscale((0.0, 1))),  # put grayscale
        ], random_order=True)  # do all of the above in random order

        return augment.augment_images(images.reshape(-1, 180, 320, 3).astype(np.uint8)) \
            .reshape(batchSize, self._seq_len, 180, 320, 3).astype(np.float32)

    def generate_train_batch(self, batch_size):

        train_targets = f['targets'][self._validation_set_size: f['targets'].shape[0]]
        train_input_front = f['Camera/RGB_front'][self._validation_set_size: f['targets'].shape[0]]
        sequence_indices = []

        for index in range(train_targets.shape[0] - self._seq_len):
            flag = True
            for i in range(self._seq_len):
                if train_targets[index + i, 20] > train_targets[index + i + 1, 20]:
                    flag = False
            if flag:
                sequence_indices.append([index + j for j in range(self._seq_len)])

        epoch = 0

        while True:
            epoch += 1
            random.shuffle(sequence_indices)
            index = 0
            for num_step in range(len(sequence_indices) // batch_size):
                train_batch_targets = train_targets[sequence_indices[index:index + batch_size], :][:, -1, :]
                train_batch_inputs = train_input_front[sequence_indices
                                                       [index:index + batch_size], :, :, :]

                if self._augmentation:
                    if random.uniform(0, 1) < 0.70:
                        train_batch_inputs = \
                            self.apply_augmentation(train_batch_inputs, batch_size)
                train_batch_inputs = np.reshape(np.transpose(train_batch_inputs, axes=[0, 2, 3, 1, 4]),
                                                [batch_size, 180, 320, 3*self._seq_len])
                index = index + batch_size

                yield train_batch_inputs / 225.0, train_batch_targets, epoch, index

    def generate_train_batch_sample(self, batch_size, sample_size):

        train_targets = f['targets'][self._validation_set_size: self._validation_set_size + sample_size]
        train_input_front = f['Camera/RGB_front'][self._validation_set_size: self._validation_set_size + sample_size]
        sequence_indices = []

        for index in range(train_targets.shape[0] - self._seq_len):
            flag = True
            for i in range(self._seq_len):
                if train_targets[index + i, 20] > train_targets[index + i + 1, 20]:
                    flag = False
            if flag:
                sequence_indices.append([index + j for j in range(self._seq_len)])

        epoch = 0

        while True:
            epoch += 1
            random.shuffle(sequence_indices)
            index = 0
            for num_step in range(len(sequence_indices) // batch_size):
                train_batch_targets = train_targets[sequence_indices[index:index + batch_size], :][:, -1, :]
                train_batch_inputs = train_input_front[sequence_indices
                                                       [index:index + batch_size], :, :, :]

                if self._augmentation:
                    if random.uniform(0, 1) < 0.7:
                        train_batch_inputs = \
                            self.apply_augmentation(train_batch_inputs, batch_size)
                train_batch_inputs = np.reshape(np.transpose(train_batch_inputs, axes=[0, 2, 3, 1, 4]),
                                                [batch_size, 180, 320, 3 * self._seq_len])
                index = index + batch_size

                yield train_batch_inputs / 225.0, train_batch_targets, epoch, index

    def generate_val_batch(self, batch_size):

        val_targets = f['targets'][0: self._validation_set_size]
        val_input_front = f['Camera/RGB_front'][0: self._validation_set_size]
        sequence_indices = []

        for index in range(val_targets.shape[0] - self._seq_len):
            flag = True
            for i in range(self._seq_len):
                if val_targets[index + i, 20] > val_targets[index + i + 1, 20]:
                    flag = False
            if flag:
                sequence_indices.append([index + j for j in range(self._seq_len)])

        epoch = 0

        while True:
            epoch += 1
            random.shuffle(sequence_indices)

            index = 0
            for num_step in range(len(sequence_indices) // batch_size):
                val_batch_targets = val_targets[sequence_indices[index:index + batch_size], :][:, -1, :]
                val_batch_inputs = val_input_front[sequence_indices
                                                   [index:index + batch_size], :, :, :]

                val_batch_inputs = np.reshape(np.transpose(val_batch_inputs, axes=[0, 2, 3, 1, 4]),
                                                [batch_size, 180, 320, 3 * self._seq_len])

                index = index + batch_size

                yield val_batch_inputs / 225.0, val_batch_targets, epoch, index

#
# test case:
#
# D = Dataset_generators()
# train = D.generate_train_batch_sample(64, 1000)
# for i in range(40):
#     trainX, trainY, ep, ind = train.__next__()
#     # trainX = np.reshape(np.transpose(trainX, axes=[0, 2, 3, 1, 4]), [64, 180, 320, 18])
#     print(i, 'epoch:', ep)
#     for j in range(6):
#
#         plot_im = trainX[0, :, :, j*3:j*3 + 3]
#         plt.imshow(plot_im)
#         plt.show()