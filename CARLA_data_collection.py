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

sys.path.insert(0, '/mnt/AshishMehta/carla8/carla/PythonClient')

import argparse
import logging
import random
import time

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

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

import h5py
from carla.planner.planner import Planner
import math
import glob
import os
import json

WINDOW_WIDTH = 320
WINDOW_HEIGHT = 180
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180
NOISE_PROB = [random.uniform(0.001, 0.01), random.uniform(0.001, 0.01), random.uniform(0.001, 0.01)]
weather_choice = random.choice([1])


def make_carla_settings():
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=120,
        NumberOfPedestrians=140,
        WeatherId=weather_choice,
        SeedVehicles=random.randint(0, 123456),
        SeedPedestrians=random.randint(0, 123456))

    # settings.randomize_seeds()
    camera0 = sensor.Camera('CameraFront')
    camera0.set(FOV=100)
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(1.35, 0, 1.40)
    camera0.set_rotation(-15.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraLeft')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(1.30, -0.50, 1.40)
    camera1.set_rotation(-20.0, 300.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('CameraRight')
    camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera2.set_position(1.30, 0.50, 1.40)
    camera2.set_rotation(-20.0, 60.0, 0.0)
    settings.add_sensor(camera2)

    # camera3 = sensor.Camera('CameraFrontDepth', PostProcessing='Depth')
    # camera3.set(FOV=100)
    # camera3.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    # camera3.set_position(135, 0, 140)
    # camera3.set_rotation(-15.0, 0.0, 0.0)
    # settings.add_sensor(camera3)
    # camera4 = sensor.Camera('CameraLeftDepth', PostProcessing='Depth')
    # camera4.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera4.set_position(130, -50, 140)
    # camera4.set_rotation(-20.0, 300.0, 0.0)
    # settings.add_sensor(camera4)
    # camera5 = sensor.Camera('CameraRightDepth', PostProcessing='Depth')
    # camera5.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera5.set_position(130, 50, 140)
    # camera5.set_rotation(-20.0, 60.0, 0.0)
    # settings.add_sensor(camera5)

    # camera6 = sensor.Camera('CameraFrontSeg', PostProcessing='SemanticSegmentation')
    # camera6.set(FOV=100)
    # camera6.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    # camera6.set_position(135, 0, 140)
    # camera6.set_rotation(-15.0, 0.0, 0.0)
    # settings.add_sensor(camera6)
    # camera7 = sensor.Camera('CameraLeftSeg', PostProcessing='SemanticSegmentation')
    # camera7.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera7.set_position(130, -50, 140)
    # camera7.set_rotation(-20.0, 300.0, 0.0)
    # settings.add_sensor(camera7)
    # camera8 = sensor.Camera('CameraRightSeg', PostProcessing='SemanticSegmentation')
    # camera8.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera8.set_position(130, 50, 140)
    # camera8.set_rotation(-20.0, 60.0, 0.0)
    # settings.add_sensor(camera8)

    return settings


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
    def __init__(self, carla_client, planner_obj, map_name, city_name, save_data, is_noisy, collection_mode):
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
        # self._map_view = self._map.get_map(WINDOW_HEIGHT) if map_name is not None else None
        self._position = None
        self._agent_positions = None
        self.planner_obj = planner_obj
        self.is_noisy = is_noisy
        self.collection_mode = collection_mode

    def execute(self, dset_target, dset_im, json_folder):
        """Launch the PyGame."""

        try:
            while True:
                if self._city_name == 'Town01':
                    poses = self._poses_town01_random()
                pygame.init()
                positions = self._initialize_game(poses)

                if self.collection_mode == 'steering':
                    numAxes = self.js.get_numaxes()
                    jsInputs = [float(self.js.get_axis(i)) for i in range(numAxes)]
                    while jsInputs[1] == 0.0:
                        print("Press brake to continue")
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                        numAxes = self.js.get_numaxes()
                        jsInputs = [float(self.js.get_axis(i)) for i in range(numAxes)]
                    while jsInputs[2] == 0.0:
                        print("Press accelerator to continue")
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                        numAxes = self.js.get_numaxes()
                        jsInputs = [float(self.js.get_axis(i)) for i in range(numAxes)]

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
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                        distance = self._on_loop(dset_target, dset_im, target, json_folder)
                        self._on_render()
                        if distance < 200.0:
                            success = True
        finally:
            pygame.quit()

    def _initialize_game(self, poses):
        if self._map_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int(
                    (WINDOW_HEIGHT / float(self._map.map_image.shape[0])) * self._map.map_image.shape[1]),
                 WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        if self.collection_mode == 'steering':
            # Adding Joystick controls here
            self.js = pygame.joystick.Joystick(0)
            self.js.init()
            axis = self.js.get_axis(1)
            jsInit = self.js.get_init()
            jsId = self.js.get_id()
            print("Joystick ID: %d Init status: %s Axis(1): %d" % (jsId, jsInit, axis))

        logging.debug('pygame started')
        return self._on_new_episode(poses)

    def _on_new_episode(self, poses):
        scene = self.client.load_settings(make_carla_settings())
        # number_of_player_starts = len(scene.player_start_spots)

        positions = scene.player_start_spots

        player_start = poses[0][0]

        print('Starting new episode at position...', player_start)
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False
        self.noise_generator = self._get_noise()

        return positions

    def _on_loop(self, dset_target, dset_im, target, json_folder):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()
        # print(measurements)

        direction = self.planner_obj.get_next_command(
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

        # self._main_image = sensor_data['CameraFront']
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
        if self.collection_mode == 'steering':
            control = self._get_steering_control()
        elif self.collection_mode == 'keyboard':
            control = self._get_keyboard_control(pygame.key.get_pressed())
        elif self.collection_mode == 'autopilot':
            control = self._get_autopilot_control(measurements)

        if self.is_noisy:
            noise, noise_flag = self.noise_generator.__next__()
            # print(noise)
        else:
            noise = [0, 0, 0]
            noise_flag = [False, False, False]

        # Set the player position
        if self._map_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        self.save_data(measurements, sensor_data, direction, control, dset_target, dset_im, noise, noise_flag,
                       json_folder)

        if self.is_noisy:
            control.steer = control.steer + noise[0]
            control.throttle = control.throttle + noise[1]
            control.brake = control.brake + noise[2]

            if control.steer > 1:
                control.steer = 1
            elif control.steer < -1:
                control.steer = -1

            if control.throttle > 1:
                control.throttle = 1
            elif control.throttle < 0:
                control.throttle = 0

            if control.brake > 1:
                control.brake = 1
            elif control.brake < 0:
                control.brake = 0

        # if control is None:
        #     self._on_new_episode(poses)
        # else:
        self.client.send_control(control)
        return distance

    @staticmethod
    def get_data_json(measurements):
        """
        Writes the measurements for saving onto the disk in json format

        :return: A dictionary for storing the measurements in json format
        """

        npc_data = []

        for agent in measurements.non_player_agents:
            if agent.HasField('vehicle'):
                npc_data.append({"type": "vehicle",
                                 "id": agent.id,
                                 "location": {"x": agent.vehicle.transform.location.x,
                                              "y": agent.vehicle.transform.location.y,
                                              "z": agent.vehicle.transform.location.z},
                                 "rotation": {"pitch": agent.vehicle.transform.rotation.pitch,
                                              "roll": agent.vehicle.transform.rotation.roll,
                                              "yaw": agent.vehicle.transform.rotation.yaw},
                                 # "acceleration": {"x": agent.vehicle.acceleration.x,
                                 #                  "y": agent.vehicle.acceleration.y,
                                 #                  "z": agent.vehicle.acceleration.z},
                                 "box_location": {"x": agent.vehicle.bounding_box.transform.location.x,
                                                  "y": agent.vehicle.bounding_box.transform.location.y,
                                                  "z": agent.vehicle.bounding_box.transform.location.z},
                                 "box_rotation": {"roll": agent.vehicle.bounding_box.transform.rotation.roll,
                                                  "pitch": agent.vehicle.bounding_box.transform.rotation.pitch,
                                                  "yaw": agent.vehicle.bounding_box.transform.rotation.yaw},
                                 "box_extent": {"x": agent.vehicle.bounding_box.extent.x,
                                                "y": agent.vehicle.bounding_box.extent.y,
                                                "z": agent.vehicle.bounding_box.extent.z},
                                 "forward_speed": agent.vehicle.forward_speed
                                 })

            elif agent.HasField('pedestrian'):
                npc_data.append({"type": "pedestrian",
                                 "id": agent.id,
                                 "location": {"x": agent.pedestrian.transform.location.x,
                                              "y": agent.pedestrian.transform.location.y,
                                              "z": agent.pedestrian.transform.location.z},
                                 "rotation": {"pitch": agent.pedestrian.transform.rotation.pitch,
                                              "roll": agent.pedestrian.transform.rotation.roll,
                                              "yaw": agent.pedestrian.transform.rotation.yaw},
                                 # "acceleration": {"x": agent.pedestrian.acceleration.x,
                                 #                  "y": agent.pedestrian.acceleration.y,
                                 #                  "z": agent.pedestrian.acceleration.z},
                                 "box_location": {"x": agent.pedestrian.bounding_box.transform.location.x,
                                                  "y": agent.pedestrian.bounding_box.transform.location.y,
                                                  "z": agent.pedestrian.bounding_box.transform.location.z},
                                 "box_rotation": {"roll": agent.pedestrian.bounding_box.transform.rotation.roll,
                                                  "pitch": agent.pedestrian.bounding_box.transform.rotation.pitch,
                                                  "yaw": agent.pedestrian.bounding_box.transform.rotation.yaw},
                                 "box_extent": {"x": agent.pedestrian.bounding_box.extent.x,
                                                "y": agent.pedestrian.bounding_box.extent.y,
                                                "z": agent.pedestrian.bounding_box.extent.z},
                                 "forward_speed": agent.pedestrian.forward_speed
                                 })
            elif agent.HasField('traffic_light'):
                npc_data.append({"type": "traffic_light",
                                 "id": agent.id,
                                 "location": {"x": agent.traffic_light.transform.location.x,
                                              "y": agent.traffic_light.transform.location.y,
                                              "z": agent.traffic_light.transform.location.z},
                                 "rotation": {"pitch": agent.traffic_light.transform.rotation.pitch,
                                              "roll": agent.traffic_light.transform.rotation.roll,
                                              "yaw": agent.traffic_light.transform.rotation.yaw},
                                 "state": agent.traffic_light.state
                                 })

            elif agent.HasField('speed_limit_sign'):
                npc_data.append({"type": "speed_limit_sign",
                                 "id": agent.id,
                                 "location": {"x": agent.speed_limit_sign.transform.location.x,
                                              "y": agent.speed_limit_sign.transform.location.y,
                                              "z": agent.speed_limit_sign.transform.location.z},
                                 "rotation": {"pitch": agent.speed_limit_sign.transform.rotation.pitch,
                                              "roll": agent.speed_limit_sign.transform.rotation.roll,
                                              "yaw": agent.speed_limit_sign.transform.rotation.yaw},
                                 "speed_limit": agent.speed_limit_sign.speed_limit
                                 })

        data = {"TimeStamps": {"game_timestamp": measurements.game_timestamp,
                               "platform_timestamp": measurements.platform_timestamp},
                "PlayerMeasurements": {"location": {
                    "x": measurements.player_measurements.transform.location.x,
                    "y": measurements.player_measurements.transform.location.y,
                    "z": measurements.player_measurements.transform.location.z},
                    "rotation": {
                        "pitch": measurements.player_measurements.transform.rotation.pitch,
                        "roll": measurements.player_measurements.transform.rotation.roll,
                        "yaw": measurements.player_measurements.transform.rotation.yaw},
                    "acceleration": {
                        "x": measurements.player_measurements.acceleration.x,
                        "y": measurements.player_measurements.acceleration.y,
                        "z": measurements.player_measurements.acceleration.z},
                    "control": {
                        "steer": measurements.player_measurements.autopilot_control.steer,
                        "throttle": measurements.player_measurements.autopilot_control.throttle,
                        "brake": measurements.player_measurements.autopilot_control.brake,
                        "hand_brake": measurements.player_measurements.autopilot_control.hand_brake,
                        "reverse": measurements.player_measurements.autopilot_control.reverse},
                    "forward_speed": measurements.player_measurements.forward_speed,
                    "collision_vehicles": measurements.player_measurements.collision_vehicles,
                    "collision_pedestrians": measurements.player_measurements.collision_pedestrians,
                    "collision_other": measurements.player_measurements.collision_other,
                    "intersection_otherlane": measurements.player_measurements.intersection_otherlane,
                    "intersection_offroad": measurements.player_measurements.intersection_offroad,
                    "box_location": {"x": measurements.player_measurements.bounding_box.transform.location.x,
                                     "y": measurements.player_measurements.bounding_box.transform.location.y,
                                     "z": measurements.player_measurements.bounding_box.transform.location.z},
                    "box_rotation": {"roll": measurements.player_measurements.bounding_box.transform.rotation.roll,
                                     "pitch": measurements.player_measurements.bounding_box.transform.rotation.pitch,
                                     "yaw": measurements.player_measurements.bounding_box.transform.rotation.yaw},
                    "box_extent": {"x": measurements.player_measurements.bounding_box.extent.x,
                                   "y": measurements.player_measurements.bounding_box.extent.y,
                                   "z": measurements.player_measurements.bounding_box.extent.z}},
                "NpcMeasurements": npc_data}
        return data

    def save_data(self, measurements, sensor_data, direction, control, dset_target, dset_im, noise, noise_flag,
                  json_folder):
        """
        0 demontstrated steering command
        1 demonstrated throttle command
        2 demonstrated brake command
        3 deonstrated hand brake command
        4 demonstrated reverse command
        5 added steering nosie
        6 added throtle noise
        7 added brake noise
        8 position x of player
        9 position y of player
        10 forward speed of player
        11 collision with static objects
        12 collision with pedestrians
        13 collision with vehicles
        14 percentage player in opposite lane
        15 percentage player on side_walk
        16 accelelration x of player
        17 accelelration y of player
        18 accelelration z of player
        19 platform time
        20 game time
        21 roll
        22 pitch
        23 yaw
        24 high-level planner command (2 Follow lane, 3 Left, 4 Right, 5 Straight)
        25 action abstractions
        26 visual abstractions
        27 noise flag
        28 autopilot steer
        29 autopilot throttle
        30 autopilot brake
        31 autopilot handbrake
        32 autopilot reverse
        33 weather ID
        """

        if self._save_data:
            steer_noise = noise[0]
            gas_noise = noise[1]
            brake_noise = noise[2]
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
            roll = measurements.player_measurements.transform.rotation.roll
            pitch = measurements.player_measurements.transform.rotation.pitch
            yaw = measurements.player_measurements.transform.rotation.yaw
            highlevel_command = direction
            abstract_action = 0
            abstract_visual = 0
            noise = 1 if noise_flag[0] == True or noise_flag[1] == True or noise_flag[2] == True else 0
            autopilot_steer = measurements.player_measurements.autopilot_control.steer
            autopilot_throttle = measurements.player_measurements.autopilot_control.throttle
            autopilot_brake = measurements.player_measurements.autopilot_control.brake
            autopilot_hand_brake = measurements.player_measurements.autopilot_control.hand_brake
            autopilot_reverse = measurements.player_measurements.autopilot_control.reverse
            weatherID = weather_choice

            target_array = np.array([control.steer, control.throttle, control.brake, control.hand_brake,
                                     control.reverse, steer_noise, gas_noise, brake_noise, pos_x, pos_y, forward_speed,
                                     collision_other, collision_pedestrians, collision_vehicles, opposite_lane,
                                     side_walk, accx, accy, accz, platform_time, game_time, roll, pitch, yaw,
                                     highlevel_command, abstract_action, abstract_visual, noise,
                                     autopilot_steer, autopilot_throttle, autopilot_brake, autopilot_hand_brake,
                                     autopilot_reverse, weatherID])

            dset_target[-1, :] = target_array
            dset_target.resize(dset_target.shape[0] + 1, axis=0)

            dset_im[0][-1, :, :, :] = sensor_data['CameraFront'].data
            dset_im[0].resize(dset_im[0].shape[0] + 1, axis=0)

            dset_im[1][-1, :, :, :] = sensor_data['CameraLeft'].data
            dset_im[1].resize(dset_im[1].shape[0] + 1, axis=0)

            dset_im[2][-1, :, :, :] = sensor_data['CameraRight'].data
            dset_im[2].resize(dset_im[2].shape[0] + 1, axis=0)

            with open(json_folder + 'measurements{}.json'.format(self._timer.step - 1), 'w') as fp:
                json.dump(self.get_data_json(measurements), fp, indent=4)

    def _get_steering_control(self):
        numAxes = self.js.get_numaxes()
        jsInputs = [float(self.js.get_axis(i)) for i in range(numAxes)]
        # print('Js inputs [%s]' % ', '.join(map(str, jsInputs)))
        control = VehicleControl()
        control.steer = jsInputs[0]
        if ((1 - jsInputs[1]) / 2) > 0.001:
            control.brake = (1 - jsInputs[1]) / 2
        else:
            control.brake = 0
        if ((1 - jsInputs[2]) / 2) > 0.001:
            control.throttle = (1 - jsInputs[2]) / 2
        else:
            control.throttle = 0
        control.hand_brake = 0.0
        control.reverse = 0.0
        # print(control)
        return control

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        control.reverse = self._is_on_reverse
        return control

    def _get_autopilot_control(self, measurements):
        control = VehicleControl()

        control.steer = measurements.player_measurements.autopilot_control.steer
        control.throttle = measurements.player_measurements.autopilot_control.throttle
        control.brake = measurements.player_measurements.autopilot_control.brake
        control.hand_brake = measurements.player_measurements.autopilot_control.hand_brake
        control.reverse = measurements.player_measurements.autopilot_control.reverse
        return control

    @staticmethod
    def _get_noise():
        pos_neg_random = [random.uniform(0, 1) for _ in range(3)]
        noise_flag = [False for _ in range(3)]
        noise = [0.0 for _ in range(3)]
        noise_counter = [0, 0, 0]
        max_count_noise = [10, 20, 20]
        while True:
            rand_array = [random.uniform(0, 1) for _ in range(3)]
            for pos, num in enumerate(rand_array):
                if num > NOISE_PROB[pos] and noise_flag[pos] == False:
                    noise[pos] = 0.0
                    noise_flag[pos] = False
                    pos_neg_random[pos] = random.uniform(0, 1)
                    max_count_noise = [random.randint(5, 15), random.randint(10, 40), random.randint(10, 40)]

                else:
                    noise_flag[pos] = True
                    noise_counter[pos] = noise_counter[pos] + 1
                    if noise_counter[pos] <= max_count_noise[pos]:
                        if pos_neg_random[pos] > 0.5:
                            noise[pos] += 0.1
                        else:
                            noise[pos] -= 0.1
                    else:
                        noise_flag[pos] = False
                        noise[pos] = 0.0
                        noise_counter[pos] = 0
            yield noise, noise_flag

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

    def _on_render(self):
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        if self._mini_view_image1 is not None:
            array = image_converter.to_rgb_array(self._mini_view_image1)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (gap_x, mini_image_y))

        if self._mini_view_image2 is not None:
            array = image_converter.to_rgb_array(
                self._mini_view_image2)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            self._display.blit(
                surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0] * (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
            h_pos = int(self._position[1] * (new_window_width / float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])

                    w_pos = int(agent_position[0] * (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
                    h_pos = int(agent_position[1] * (new_window_width / float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()
        # pass

    def _poses_town01(self):

        return [[110, 111], [111, 4]]

    def _poses_town01_random(self):

        poses = []
        rand_poses = [random.randint(1, 140) for _ in range(100)]
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
        '-n', '--enable-noise',
        action='store_true',
        help='Add noise to player commands'
    )
    argparser.add_argument(
        '-mo', '--collection-mode',
        default='keyboard',
        help='For data collection: keyboard = use keyboard, steering = use steering wheel, auto = use autopilot'
    )
    argparser.add_argument(
        '-s', '--save-data',
        action='store_true',
        help='Store the demonstrated data to disk')
    argparser.add_argument(
        '-l', '--player-name',
        metavar='L',
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
        file_count = int(file_count / 2)
        f = h5py.File("./Data_Set/" + args.player_name + "/Training_Data{}.h5".format(file_count), "w")
        if not os.path.exists('./Data_Set/' + args.player_name + '/Training_Data{}/'.format(file_count)):
            os.makedirs('./Data_Set/' + args.player_name + '/Training_Data{}/'.format(file_count))
        json_folder = './Data_Set/' + args.player_name + '/Training_Data{}/'.format(file_count)

        dset_target = f.create_dataset('targets', (1, 34), maxshape=(None, 50), dtype=np.float128)
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
        json_folder = None

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                planner_obj = Planner(args.city_name)
                game = CarlaGame(client, planner_obj, args.map_name, args.city_name, args.save_data, args.enable_noise,
                                 args.collection_mode)
                game.execute(dset_target, dset_im, json_folder)
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:

        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
