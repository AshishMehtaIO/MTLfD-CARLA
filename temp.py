# import json
#
#
# dir = '/mnt/AshishMehta/CARLA_EndTOEndDriving/Data_Set/Ashish_Noisy/Training_Data0.0/'
#
#
# for i in range(1200):
#     measurements = json.load(open(dir + 'measurements{}.json'.format(i)))
#     player_pos_x = measurements['PlayerMeasurements']['location']['x']
#     player_pos_y = measurements['PlayerMeasurements']['location']['y']
#     player_yaw = measurements['PlayerMeasurements']['rotation']['yaw']
#     print(player_pos_x, '\t', player_pos_y, '\t', player_yaw)
#
#
#
# # for
# # measurements['NpcMeasurements'][1]['type']
#
# # player_speed = measurements['PlayerMeasurements']['forward_speed']
# # player_intersection_otherlane = measurements['PlayerMeasurements']['intersection_otherlane']
# # player_intersection_sidewalk = measurements['PlayerMeasurements']['intersection_offroad']
# # player_yaw = measurements['PlayerMeasurements']['rotation']['yaw']
# # player_pos_x = measurements['PlayerMeasurements']['location']['x']
# # player_pos_y = measurements['PlayerMeasurements']['location']['y']
#
#
# '''
# "demontstrated_steering_command"
#     "demonstrated_throttle_command"
#     "demonstrated_brake_command"
#     "deonstrated_hand_brake_command"
#     "demonstrated_reverse_command"
#     "added_steering_nosie"
#     "added_throtle_noise"
#     "added_brake_noise"
#     "player_position_x"
#     "player_position_y"
#     "player_forward_speed"
#     "collision_static_objects"
#     "collision_pedestrians"
#     "collision_vehicles"
#     "percentage_player_opposite_lane"
#     "percentage_player_side_walk"
#     "player_accelelration_x"
#     "player_accelelration_y"
#     "player_accelelration_z"
#     "platform_time"
#     "game_time"
#     "roll"
#     "pitch"
#     "yaw"
#     "planner_command"
#     "action_abstractions"
#     'visual_abstractions"
#     "noise_flag"
#     "autopilot_steer"
#     "autopilot_throttle"
#     "autopilot_brake"
#     "autopilot_handbrake"
#     "autopilot_reverse"
#     "weather_ID"'''


import tensorflow as tf


class Network(object):

    def __init__(self, dropout, training_mode):
        self._conv_count = 0
        self._dense_count = 0
        self._dropout_count = 0
        self._dropout_vec = dropout
        self._training_mode = training_mode

    def conv_block(self, features, kernel_size, strides, filters):
        self._conv_count = self._conv_count + 1
        self._dropout_count = self._dropout_count + 1

        # Convolutional Layer #1
        conv = tf.layers.conv2d(
            inputs=features,
            filters=filters,
            strides=(strides, strides),
            kernel_size=[kernel_size, kernel_size],
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=True,
            padding="valid",
            activation=None,
            name='Conv_{}'.format(self._conv_count),
            reuse=True)

        batch_norm = tf.layers.batch_normalization(
            inputs=conv,
            trainable=True,
            training=self._training_mode,
            name='BatchNorm_{}'.format(self._conv_count),
            reuse='True')

        dropout = tf.layers.dropout(
            inputs=batch_norm,
            rate=self._dropout_vec[self._dropout_count],
            training=self._training_mode,
            name='Conv_dropout_{}'.format(self._conv_count))

        return tf.nn.relu(dropout)

    def dense_block(self, features, output_size):
        self._dense_count = self._dense_count + 1
        self._dropout_count = self._dropout_count + 1

        dense = tf.layers.dense(
            inputs=features,
            units=output_size,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='Dense_{}'.format(self._dense_count),
            reuse=True)

        dropout = tf.layers.dropout(
            inputs=dense,
            rate=self._dropout_vec[self._dropout_count],
            name='Dense_dropout_{}'.format(self._dense_count),
            training=self._training_mode)

        return tf.nn.relu(dropout)

    def CNN_feature_extractor(self, input_im, speed):

        """conv1"""  # kernel sz, stride, num feature maps
        xc = self.conv_block(input_im, 5, 2, 32)
        print(xc)
        xc = self.conv_block(xc, 3, 1, 32)
        print(xc)

        """conv2"""
        xc = self.conv_block(xc, 3, 2, 64)
        print(xc)
        xc = self.conv_block(xc, 3, 1, 64)
        print(xc)

        """conv3"""
        xc = self.conv_block(xc, 3, 2, 128)
        print(xc)
        xc = self.conv_block(xc, 3, 1, 128)
        print(xc)

        """conv4"""
        xc = self.conv_block(xc, 3, 1, 256)
        print(xc)
        xc = self.conv_block(xc, 3, 1, 256)
        print(xc)

        """ reshape """
        x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
        print(x)

        """ fc1 """
        x = self.dense_block(x, 512)
        print(x)
        """ fc2 """
        x = self.dense_block(x, 512)
        print(x)

        """ Speed (measurements)"""
        with tf.variable_scope("Speed"):
            """ fc3 """
            speed_net = self.dense_block(speed, 128)
            """ fc4 """
            speed_net = self.dense_block(speed_net, 128)

        # with tf.variable_scope("HighLevelPlanner"):
        #     """ fc5 """
        #     plan_net = self.fc_block(planner, 128)
        #     """ fc6 """
        #     plan_net = self.fc_block(plan_net, 128)

        """ Joint sensory """
        # j = tf.concat([x, speed_net, plan_net], 1)
        j = tf.concat([x, speed_net], 1)

        """ fc7 """
        j = self.dense_block(j, 512)

        j = self.dense_block(j, 512)

        j = self.dense_block(j, 512)

        """ fc8 """
        out = self.dense_block(j, 512)

        self._conv_count = 0
        self._dense_count = 0
        self._dropout_count = 0

        return out




