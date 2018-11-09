import numpy as np
import tensorflow as tf
from load_sequential_CNN_dataset import Dataset_generators
import CNN_Network as N
import argparse
from pygit2 import Repository
import logging
import sys
import os


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets, dtype=np.int8)].reshape(targets.shape[0], nb_classes)


def sparse_action_visual_abstraction(action):
    sparse_action = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(len(action))]
    for i in range(len(action)):
        if int(action[i]) & 0b0000000000000010:
            sparse_action[i][1] = 1
        if int(action[i]) & 0b0000000000000001:
            sparse_action[i][0] = 1
        if int(action[i]) & 0b0000000000000100:
            sparse_action[i][2] = 1
        if int(action[i]) & 0b0000000000001000:
            sparse_action[i][3] = 1
        if int(action[i]) & 0b0000000000010000:
            sparse_action[i][4] = 1
        if int(action[i]) & 0b0000000000100000:
            sparse_action[i][5] = 1
        if int(action[i]) & 0b0000000001000000:
            sparse_action[i][6] = 1
        if int(action[i]) & 0b0000000010000000:
            sparse_action[i][7] = 1

    return sparse_action


def main():
    argparser = argparse.ArgumentParser(
        description='Training Script')
    argparser.add_argument(
        '--max-epoch',
        default=500,
        type=int,
        help='Max number of epochs (default = 500)')
    argparser.add_argument(
        '--batch-size',
        default=24,
        type=int,
        help='Batch size (default = 64)')
    argparser.add_argument(
        '--learning-rate',
        default=1e-5,
        type=float,
        help='Learning rate (default = 1e-5)')
    argparser.add_argument(
        '--load-network',
        action='store_true',
        default=False,
        help='Load previously stored network')
    argparser.add_argument(
        '--checkpoint-restore',
        default=10,
        type=int,
        help='Checkpoint number from which the model has to be restored (default = 10)')
    argparser.add_argument(
        '--restore-dir',
        default='exp1',
        help='Directory from which to restore checkpoint (default = exp1)')
    argparser.add_argument(
        '--sample-train',
        action='store_true',
        default=False,
        help='Load only sample training set into memory'
    )
    argparser.add_argument(
        '--max-checkpoints',
        default=20,
        type=int,
        help='Max number of checkpoints to be stored to disk (default = 20)')
    argparser.add_argument(
        '--save-dir',
        default='exp1',
        help='Directory to save learnt models (default = exp1)'
    )
    argparser.add_argument(
        '--checkpoint-frequency',
        default=2,
        type=int,
        help='Checkpoint after every n epochs(default =2)'
    )
    argparser.add_argument(
        '--decay-rate',
        default=0.99995,
        type=float,
        help='Learning Rate Decay (default=0.99995)'
    )
    argparser.add_argument(
        '--hist-timesteps',
        default=6,
        type=int,
        help='Number of times steps to unroll the RNN into the past (default=6)'
    )

    argparser.add_argument(
        '--future-timesteps',
        default=10,
        type=int,
        help='Number of times steps to unroll the RNN into the future (default=10)'
    )
    argparser.add_argument(
        '--use-cpu',
        action='store_true',
        default=False,
        help='Use CPU for computation'
    )
    argparser.add_argument(
        '--alpha',
        default=0.1,
        type=float,
        help='Abstraction auxiliary loss coefficient (default = 0.1)'
    )
    argparser.add_argument(
        '--beta',
        default=0.2,
        type=float,
        help='Multitask Driving auxiliary loss coefficient (default = 0.2)'
    )
    argparser.add_argument(
        '--gamma',
        default=1e-5,
        type=float,
        help='Multitask Driving auxiliary loss coefficient (default = 1e-5)'
    )
    argparser.add_argument(
        '--dropout-rate',
        default=0.6,
        type=float,
        help='Dropout rate (default = 0.6)'
    )

    args = argparser.parse_args()

    branch_name = Repository('.').head.shorthand
    epochs = args.max_epoch
    batchSize = args.batch_size
    init_LR = args.learning_rate
    load_network = args.load_network
    RESTORE_DIR = './saved_data/' + branch_name + '/' + args.restore_dir + \
                  '/model_ep{}.ckpt'.format(args.checkpoint_restore)
    SAVE_DIR = './saved_data/' + branch_name + '/' + args.save_dir
    checkpoint_frequency = args.checkpoint_frequency
    lr_decay = args.decay_rate

    if args.use_cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(args)

    # Initialize variables
    step_in_epoch_train = 0
    step_in_epoch_val = 0
    step = 0
    prev_epoch = 0
    prev_val_epoch = 0
    epoch = 1

    total_train_loss = 0
    avg_train_loss = 0
    total_val_loss = 0
    avg_val_loss = 0

    total_train_main_loss = 0
    total_val_main_loss = 0
    total_train_auxiliary_loss = 0
    total_val_auxiliary_loss = 0
    avg_train_main_loss = 0
    avg_val_main_loss = 0
    avg_train_auxiliary_loss = 0
    avg_val_auxiliary_loss = 0

    state_size = 512
    hist_timesteps = args.hist_timesteps
    future_timesteps = args.future_timesteps
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    train_init_cell_state = np.zeros(shape=[batchSize, state_size])
    train_init_hidden_state = np.zeros(shape=[batchSize, state_size])
    val_init_cell_state = np.zeros(shape=[batchSize, state_size])
    val_init_hidden_state = np.zeros(shape=[batchSize, state_size])

    D = Dataset_generators(single_image=True, augmentation=True, preload_data=True, only_lane_follow=False,
                           sequential_data=True, validation_set_size=10000, seq_len=hist_timesteps)

    val_batch = D.generate_val_batch(batchSize)

    lr = init_LR

    input_im, speed, planner, auxiliary_visual_y, auxiliary_action_y, driving_y, \
    weighted_auxiliary_losses, action_abstraction_accuracy, main_loss, loss, \
    train_op, saver, learning_rate, driving_logits = \
        N.make_network(max_checkpoints=args.max_checkpoints,
                       hist_timesteps=hist_timesteps,
                       num_channels=hist_timesteps * 3,
                       batch_size=batchSize,
                       alpha=alpha,
                       beta=beta,
                       gamma=gamma,
                       mode='TRAIN',
                       dropout_rate=args.dropout_rate)

    avg_train_loss_t = tf.Variable(0.0, name='AvgTrainLoss')
    assign_avg_train = avg_train_loss_t.assign(avg_train_loss)
    tf.summary.scalar('AvgTrainLoss', avg_train_loss_t)

    avg_val_loss_t = tf.Variable(0.0, name='AvgValLoss')
    assign_avg_val = avg_val_loss_t.assign(avg_val_loss)
    tf.summary.scalar('AvgValLoss', avg_val_loss_t)

    avg_train_main_loss_t = tf.Variable(0.0, name='AvgTrainMainLoss')
    assign_avg_train_main = avg_train_main_loss_t.assign(avg_train_main_loss)
    tf.summary.scalar('TrainMainLoss', avg_train_main_loss_t)

    avg_val_main_loss_t = tf.Variable(0.0, name='AvgValMainLoss')
    assign_avg_val_main = avg_val_main_loss_t.assign(avg_val_main_loss)
    tf.summary.scalar('ValMainLoss', avg_val_main_loss_t)

    avg_train_auxiliary_loss_t = tf.Variable(0.0, name='AvgTrainAuxLoss')
    assign_avg_train_auxiliary = avg_train_auxiliary_loss_t.assign(avg_train_auxiliary_loss)
    tf.summary.scalar('TrainAuxLoss', avg_train_auxiliary_loss_t)

    avg_val_auxiliary_loss_t = tf.Variable(0.0, name='AvgValAuxLoss')
    assign_avg_val_auxiliary = avg_val_auxiliary_loss_t.assign(avg_val_auxiliary_loss)
    tf.summary.scalar('ValAuxLoss', avg_val_auxiliary_loss_t)

    tf.summary.scalar('Auxillary_Action_Accuracy', action_abstraction_accuracy)
    # tf.summary.scalar('ValAccuracy', accuracy_val)
    tf.summary.scalar('LearningRate', learning_rate)

    merged = tf.summary.merge_all()

    sess = tf.Session()
    if load_network:
        saver.restore(sess, RESTORE_DIR)
        print("Network restored")
    else:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        print("Network initialized")

    train_writer = tf.summary.FileWriter(SAVE_DIR + '/TfBoard', sess.graph)

    while epoch <= epochs:
        step += 1
        step_in_epoch_train += 1


        step_in_epoch_val += 1

        valX, valY, val_epoch, val_index = val_batch.__next__()
        measurements_val = valY[:, 0:3]
        speed_vec_val = valY[:, 10:11] / 120
        highlevel_command_val = get_one_hot(valY[:, 24:25] - 2, 4)
        action_abstraction_val = sparse_action_visual_abstraction(valY[:, 25])
        visual_abstraction_val = valY[:, 34:47]

        assign_avg_train = avg_train_loss_t.assign(avg_train_loss)
        assign_avg_val = avg_val_loss_t.assign(avg_val_loss)
        assign_avg_train_main = avg_train_main_loss_t.assign(avg_train_main_loss)
        assign_avg_val_main = avg_val_main_loss_t.assign(avg_val_main_loss)
        assign_avg_train_auxiliary = avg_train_auxiliary_loss_t.assign(avg_train_auxiliary_loss)
        assign_avg_val_auxiliary = avg_val_auxiliary_loss_t.assign(avg_val_auxiliary_loss)

        sess.run(assign_avg_train)
        sess.run(assign_avg_val)
        sess.run(assign_avg_train_main)
        sess.run(assign_avg_val_main)
        sess.run(assign_avg_train_auxiliary)
        sess.run(assign_avg_val_auxiliary)

        val_loss, val_auxiliary_loss, val_main_loss, summary, val_commands = \
            sess.run([loss, weighted_auxiliary_losses, main_loss, merged, driving_logits],
                     feed_dict={input_im: valX,
                                auxiliary_visual_y: visual_abstraction_val,
                                auxiliary_action_y: action_abstraction_val,
                                driving_y: measurements_val,
                                speed: speed_vec_val,
                                planner: highlevel_command_val,
                                learning_rate: lr
                                })

        train_writer.add_summary(summary, step)

        total_val_loss = total_val_loss + val_loss
        total_val_auxiliary_loss = total_val_auxiliary_loss + val_auxiliary_loss
        total_val_main_loss = total_val_main_loss + val_main_loss
        avg_val_loss = total_val_loss / step_in_epoch_val
        avg_val_auxiliary_loss = total_val_auxiliary_loss / step_in_epoch_val
        avg_val_main_loss = total_val_main_loss / step_in_epoch_val

        if val_epoch > prev_val_epoch:
            step_in_epoch_val = 0
            total_val_loss = 0
            total_val_auxiliary_loss = 0
            total_val_main_loss = 0
        #
        # print("Val Loss: ", val_loss, "\tAvgVal Loss: ", avg_val_loss,
        #       "\tVal Auxiliary Loss: ", val_auxiliary_loss, "\tAvgVal Auxiliary Loss: ",
        #       avg_val_auxiliary_loss, "\tVal Main Loss: ", val_main_loss,
        #       "\tAvgVal Main Loss: ", avg_val_main_loss, '\n')

        print(step_in_epoch_val)
        print("Val Loss: ", val_main_loss, "\tAvgVal Loss: ", avg_val_main_loss, )
        print(list([i, j] for i,j in zip(measurements_val, val_commands)))
        print('\n')

        prev_val_epoch = val_epoch

        if epoch > prev_epoch:
            step_in_epoch_train = 0
            total_train_loss = 0
            total_train_auxiliary_loss = 0
            total_train_main_loss = 0

        lr = lr * lr_decay

        if epoch > prev_epoch and epoch % checkpoint_frequency == 0:
            save_path = saver.save(sess, SAVE_DIR + '/model_ep{}.ckpt'.format(epoch))
            print("Model saved in path: %s" % save_path)
            # print('Training Output')
            #  print(np.dstack((action_abstraction_train, output_vec)))
            # print('Validation Output')
            #  print(np.dstack((action_abstraction_val, output_vec_val)))

        prev_epoch = epoch


main()


