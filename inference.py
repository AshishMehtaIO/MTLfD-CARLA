#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake

    R            : restart level

STARTING in a moment...
"""
from __future__ import print_function
import sys
import tensorflow as tf

sys.path.insert(0, '/mnt/AshishMehta/carla8/carla/PythonClient')

import argparse
import logging
import random
import time
import CNN_Network as N
from pygit2 import Repository
from collections import deque

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from carla.planner.planner import Planner

import h5py
import math
import glob
import os

WINDOW_WIDTH = 320
WINDOW_HEIGHT = 180
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

val_input_im_t, val_speed_t, val_planner_t, val_auxiliary_visual_y_t, val_auxiliary_action_y_t, val_driving_y_t, \
val_weighted_auxiliary_losses_t, val_action_abstraction_accuracy_t, val_main_loss_t, val_loss_t, \
val_train_op_t, val_saver_t, val_learning_rate_t, val_driving_logits_t = \
    N.make_network(max_checkpoints=500,
                   hist_timesteps=6,
                   num_channels=6 * 3,
                   batch_size=1,
                   alpha=0.1,
                   beta=0.2,
                   gamma=1e-5,
                   mode='VAL',
                   dropout_rate=0)
sess = tf.Session()
branch_name = Repository('.').head.shorthand

val_saver_t.restore(sess, './saved_data/' + branch_name + '/exp3' + '/model_ep10.ckpt')

im_q = deque(maxlen=6)
for i in range(6):
    im_q.append(np.zeros([180, 320, 3]))


def make_carla_settings():
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=False,
        NumberOfVehicles=120,
        NumberOfPedestrians=140,
        WeatherId=random.choice([1]),
        SeedVehicles=random.randint(0, 123456),
        SeedPedestrians=random.randint(0, 123456))

    # settings.randomize_seeds()
    camera0 = sensor.Camera('CameraFront')
    camera0.set(FOV=100)
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(135, 0, 140)
    camera0.set_rotation(-15.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraLeft')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(130, -50, 140)
    camera1.set_rotation(-20.0, 300.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('CameraRight')
    camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera2.set_position(130, 50, 140)
    camera2.set_rotation(-20.0, 60.0, 0.0)
    settings.add_sensor(camera2)
    return settings


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets, dtype=np.int8).reshape(-1)]


def get_inference(measurements, sensor_data, direction):
    im_q.append(np.asarray([sensor_data['CameraFront'].data / 255.0][0]))

    speed_vec = [[measurements.player_measurements.forward_speed / 120]]

    highlevel_command = get_one_hot(direction - 2, 4)

    feed_im = np.asarray(im_q).transpose([1, 2, 0, 3]).reshape([1, 180, 320, 18])

    control = sess.run(val_driving_logits_t, feed_dict={val_input_im_t: feed_im,
                                                        val_speed_t: speed_vec,
                                                        val_planner_t: highlevel_command
                                                        # val_auxiliary_visual_y_t: np.zeros([1, 13]),
                                                        # val_auxiliary_action_y_t: np.zeros([1, 8]),
                                                        # val_driving_y_t: np.zeros([1, 3]),
                                                        # val_learning_rate_t: [0]
                                                        })
    print("Command: ", control[0])
    return control[0]


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, planner_obj, map_name, city_name, save_data):
        self.client = carla_client
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_lidar = False
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = city_name
        self._map_name = map_name
        self._save_data = save_data
        self._map = CarlaMap(map_name, 16.43, 50.0) if map_name is not None else None
        self._map_shape = self._map.map_image.shape if map_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if map_name is not None else None
        self._position = None
        self._agent_positions = None
        self._planner_obj = planner_obj

    def execute(self, dset_target, dset_im):
        """Launch the PyGame."""

        while True:
            if self._city_name == 'Town01':
                poses = self._poses_town01_random()
            positions = self._on_new_episode(poses)

            for pose in poses:
                start_point = pose[0]
                end_point = pose[1]
                target = positions[end_point]
                for i in range(10):
                    logging.info('======================================== !!!!!!!!!!!!!!!! '
                                 '========================================')
                logging.info(' Start Position %d End Position %d ',
                             start_point, end_point)
                success = False
                while not success:
                    distance = self._on_loop(dset_target, dset_im, target)
                    if distance < 200.0:
                        success = True

    def _on_new_episode(self, poses):
        scene = self.client.load_settings(make_carla_settings())
        # number_of_player_starts = len(scene.player_start_spots)

        positions = scene.player_start_spots

        player_start = poses[0][0]

        print('Starting new episode at position...', player_start)
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

        return positions

    def _on_loop(self, dset_target, dset_im, target):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()
        # print(measurements)

        direction = self._planner_obj.get_next_command(
            (measurements.player_measurements.transform.location.x,
             measurements.player_measurements.transform.location.y, 22),
            (measurements.player_measurements.transform.orientation.x,
             measurements.player_measurements.transform.orientation.y,
             measurements.player_measurements.transform.orientation.z),
            (target.location.x, target.location.y, 22),
            (target.orientation.x, target.orientation.y, -0.001))

        curr_x = measurements.player_measurements.transform.location.x
        curr_y = measurements.player_measurements.transform.location.y

        # print(curr_x, '\t', curr_y, '\t', target.location.x,'\t',target.location.y)
        distance = self.sldist([curr_x, curr_y],
                               [target.location.x, target.location.y])

        if direction == 2:
            print("Follow lane\t \t", "Distance\t", distance, "\tStep\t", self._timer.step)
        elif direction == 3:
            print("Go Left\t \t", "Distance\t", distance, "\tStep\t", self._timer.step)
        elif direction == 4:
            print("Go Right\t \t", "Distance\t", distance, "\tStep\t", self._timer.step)
        elif direction == 5:
            print("Go straight\t \t", "Distance ", distance, "\tStep ", self._timer.step)
        else:
            print("Distance ", distance, "\tStep ", self._timer.step)

        self._main_image = sensor_data['CameraFront']
        # self._mini_view_image1 = sensor_data['CameraLeft']
        # self._mini_view_image2 = sensor_data['CameraRight']

        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._map_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            else:
                self._print_player_measurements(measurements.player_measurements)

            # Plot position on the map as well.

            self._timer.lap()

        infered_control = get_inference(measurements, sensor_data, direction)
        control = VehicleControl()
        # if infered_control[0] > 0.05 or infered_control[0] < -0.05:
        #     control.steer = infered_control[0]
        # else:
        control.steer = 0

        if infered_control[1] > 0.05:
            control.throttle = infered_control[1]
        else:
            control.throttle = 0

        if infered_control[2] > 0.05 and infered_control[1] < 0.05:
            control.brake = infered_control[2]
        else:
            control.brake = 0

        control.hand_brake = 0
        control.reverse = 0
        # Set the player position
        if self._map_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        self.save_data(measurements, sensor_data, direction, control, dset_target, dset_im)

        # if control is None:
        #     self._on_new_episode(poses)
        # else:
        self.client.send_control(control)
        return distance

    def save_data(self, measurements, sensor_data, direction, control, dset_target, dset_im):
        if self._save_data:
            steer_noise = 0
            gas_noise = 0
            brake_noise = 0
            pos_x = measurements.player_measurements.transform.location.x
            pos_y = measurements.player_measurements.transform.location.y
            forward_speed = measurements.player_measurements.forward_speed
            collision_other = measurements.player_measurements.collision_other
            collision_pedestrians = measurements.player_measurements.collision_pedestrians
            collision_vehicles = measurements.player_measurements.collision_vehicles
            opposite_lane = measurements.player_measurements.intersection_otherlane
            side_walk = measurements.player_measurements.intersection_offroad
            accx = measurements.player_measurements.acceleration.x
            accy = measurements.player_measurements.acceleration.y
            accz = measurements.player_measurements.acceleration.z
            platform_time = measurements.platform_timestamp
            game_time = measurements.game_timestamp
            orientationx = measurements.player_measurements.transform.orientation.x
            orientationy = measurements.player_measurements.transform.orientation.y
            orientationz = measurements.player_measurements.transform.orientation.z
            rotation = measurements.player_measurements.transform.rotation.yaw
            highlevel_command = direction
            lowlevel_command = 0
            noise = 0
            autopilot_steer = measurements.player_measurements.autopilot_control.steer
            autopilot_throttle = measurements.player_measurements.autopilot_control.throttle
            autopilot_brake = measurements.player_measurements.autopilot_control.brake
            autopilot_hand_brake = measurements.player_measurements.autopilot_control.hand_brake
            autopilot_reverse = measurements.player_measurements.autopilot_control.reverse

            target_array = np.array([control.steer, control.throttle, control.brake, control.hand_brake,
                                     control.reverse, steer_noise, gas_noise, brake_noise, pos_x, pos_y, forward_speed,
                                     collision_other, collision_pedestrians, collision_vehicles, opposite_lane,
                                     side_walk, accx, accy, accz, platform_time, game_time, orientationx, orientationy,
                                     orientationz, rotation, highlevel_command, lowlevel_command, noise,
                                     autopilot_steer,
                                     autopilot_throttle, autopilot_brake, autopilot_hand_brake, autopilot_reverse])

            dset_target[-1, :] = target_array
            dset_target.resize(dset_target.shape[0] + 1, axis=0)

            dset_im[0][-1, :, :, :] = sensor_data['CameraFront'].data
            dset_im[0].resize(dset_im[0].shape[0] + 1, axis=0)

            dset_im[1][-1, :, :, :] = sensor_data['CameraLeft'].data
            dset_im[1].resize(dset_im[1].shape[0] + 1, axis=0)

            dset_im[2][-1, :, :, :] = sensor_data['CameraRight'].data
            dset_im[2].resize(dset_im[2].shape[0] + 1, axis=0)

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        # print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        # print_over_same_line(message)

    def _poses_town01(self):

        return [[110, 111], [111, 4]]

    def _poses_town01_random(self):

        poses = []
        rand_poses = [random.randint(1, 100) for _ in range(100)]
        mem = random.randint(1, 100)
        for i in rand_poses:
            poses.append([mem, i])
            mem = i

        return poses

    def sldist(self, c1, c2):
        return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default='Town01',
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark '
             '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-s', '--save-data',
        metavar='S',
        default=False,
        type=bool,
        help='Store the demonstrated data to disk')
    argparser.add_argument(
        '-n', '--player-name',
        metavar='N',
        default='AshishMehta',
        help='Name of the demonstrator')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    if args.save_data:
        file_count = 0
        if not os.path.exists('./Data_Set/' + args.player_name):
            os.makedirs('./Data_Set/' + args.player_name)
        for _ in glob.glob('./Data_Set/' + args.player_name + '/*'):
            file_count += 1
        f = h5py.File("./Data_Set/" + args.player_name + "/Training_Data{}.h5".format(file_count), "w")

        dset_target = f.create_dataset('targets', (1, 33), maxshape=(None, 50), dtype=np.float128)
        dset_im_front = f.create_dataset('Camera/RGB_front', (1, WINDOW_HEIGHT, WINDOW_WIDTH, 3),
                                         maxshape=(None, WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        dset_im_left = f.create_dataset('Camera/RGB_left', (1, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3),
                                        maxshape=(None, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3), dtype=np.uint8)
        dset_im_right = f.create_dataset('Camera/RGB_right', (1, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3),
                                         maxshape=(None, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3), dtype=np.uint8)
        dset_im = [dset_im_front, dset_im_left, dset_im_right]
    else:
        dset_target = None
        dset_im = None

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                planner_obj = Planner(args.city_name)
                game = CarlaGame(client, planner_obj, args.map_name, args.city_name, args.save_data)
                game.execute(dset_target, dset_im)
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:

        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
