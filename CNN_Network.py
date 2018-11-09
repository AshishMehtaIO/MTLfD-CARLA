import numpy as np
import tensorflow as tf
from resnet_model import Model


def make_network(max_checkpoints, hist_timesteps, num_channels, batch_size, alpha, beta, gamma, mode, dropout_rate):

    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):

        resnet = Model(resnet_size=50,
                       bottleneck=True,
                       num_classes=1024,
                       num_filters=64,
                       kernel_size=3,
                       conv_stride=2,
                       first_pool_size=None,
                       first_pool_stride=None,
                       block_sizes=[3, 4, 6, 3],
                       block_strides=[1, 2, 2, 2],
                       final_size=2048,
                       resnet_version=2,
                       data_format='channels_last',
                       dtype=tf.float32)

        if mode == 'TRAIN':
            training_mode = True
        else:
            training_mode = False

        input_size = [None, 180, 320,  num_channels]

        input_im = tf.placeholder(tf.float32, shape=input_size)
        speed = tf.placeholder(tf.float32, shape=[None, 1])
        planner = tf.placeholder(tf.float32, shape=[None, 4])
        auxiliary_visual_y = tf.placeholder(tf.float32, shape=[None, 13])
        auxiliary_action_y = tf.placeholder(tf.float32, shape=[None, 8])
        driving_y = tf.placeholder(tf.float32, shape=[None, 3])

        with tf.variable_scope('CNN_feature_extractor', reuse=tf.AUTO_REUSE):
            extracted_features = resnet(input_im, speed, planner, training_mode, dropout_rate)

        extracted_features = tf.layers.dense(extracted_features, 512)
        extracted_features = tf.nn.elu(extracted_features)
        visual_net = tf.layers.dense(extracted_features, 256)
        # visual_net = tf.layers.batch_normalization(inputs=visual_net, training=training_mode)
        visual_net = tf.nn.elu(visual_net)
        visual_logits = tf.layers.dense(visual_net, 13)
        visual_logits_dense = tf.add(tf.nn.elu(tf.layers.dense(visual_logits, 512)), extracted_features)
        # visual_logits_dense = tf.layers.batch_normalization(visual_logits_dense, training=training_mode)
        visual_logits_dense = tf.nn.elu(visual_logits_dense)

        action_net = tf.layers.dense(extracted_features, 256)
        # action_net = tf.layers.batch_normalization(inputs=action_net, training=training_mode)
        action_net = tf.nn.elu(action_net)
        action_logits = tf.layers.dense(action_net, 8)
        action_logits_dense = tf.add(tf.nn.elu(tf.layers.dense(action_logits, 512)), extracted_features)
        # action_logits_dense = tf.layers.batch_normalization(action_logits_dense, training=training_mode)
        action_logits_dense = tf.nn.elu(action_logits_dense)

        visual_attention_model = tf.nn.softmax(tf.layers.dense(visual_logits_dense, 512))
        visual_context = tf.multiply(extracted_features, visual_attention_model)

        action_attention_model = tf.nn.softmax(tf.layers.dense(action_logits_dense, 512))
        action_context = tf.multiply(extracted_features, action_attention_model)

        driving_net = tf.add(visual_context, action_context)

        driving_net = tf.layers.dense(driving_net, 512)
        driving_net = tf.nn.elu(driving_net)
        driving_net = tf.layers.dense(driving_net, 512)
        driving_net = tf.nn.elu(driving_net)
        # driving_net = tf.layers.batch_normalization(driving_net, training=training_mode)
        driving_net = tf.nn.elu(driving_net)
        driving_logits = tf.layers.dense(driving_net, 3)

        main_loss = tf.reduce_mean(tf.losses.mean_squared_error(driving_y, driving_logits))
        aux_action_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=auxiliary_action_y,
                                                                                 logits=action_logits))
        aux_visual_loss = tf.reduce_mean(tf.losses.mean_squared_error(auxiliary_visual_y, visual_logits))

        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])

        action_abstraction_outputs = tf.nn.sigmoid(action_logits)
        abstraction_pred = tf.cast(tf.greater(action_abstraction_outputs, 0.5), tf.float32)
        action_abstraction_accuracy = tf.reduce_mean(tf.cast(tf.equal(abstraction_pred, auxiliary_action_y), tf.float32))

        weighted_auxiliary_losses = alpha * aux_visual_loss + beta * aux_action_loss + gamma * lossL2
        loss = main_loss + weighted_auxiliary_losses
        learning_rate = tf.placeholder(tf.float32, shape=[])
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss=loss)

        saver = tf.train.Saver(max_to_keep=max_checkpoints)

        return input_im, speed, planner, auxiliary_visual_y, auxiliary_action_y, driving_y, \
            weighted_auxiliary_losses, action_abstraction_accuracy, main_loss, loss, \
            train_op, saver, learning_rate, driving_logits #, aux_visual_loss, aux_action_loss, lossL2
