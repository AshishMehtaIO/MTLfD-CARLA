import h5py
import numpy as np
import json
import pylab as pl
import glob


def check_speed_up(control, control_t1):
    if control[1] > 0.1:
        if round(control[1], 2) >= round(control_t1[1], 2):
            return True
    return False


def check_slow_down(control, control_t1):
    if 0.05 < control[2] < 0.10:
        # if round(control[2], 2) >= round(control_t1[2], 2):
        return True
    elif round(control[1], 2) < round(control_t1[1], 2):
        return True
    return False


def check_stop(control):
    if control[2] >= 0.10:
        return True
    return False


def check_do_nothing(control, speed):
    if round(speed) == 0:
        if round(control[1], 1) == 0.0:
            return True
    return False


def check_left(control):
    if control[0] < -0.05:
        return True
    return False


def check_right(control):
    if control[0] > 0.05:
        return True
    return False


def check_cut_out(control, other_lane, other_lane_t1):
    if other_lane > 0.15 and other_lane > other_lane_t1:
        return True
    return False


def check_cut_in(control, other_lane, other_lane_t1):
    if other_lane > 0.15 and other_lane_t1 > other_lane:
        return True
    return False


def check_stationary_vehicle(player_x, player_y, player_yaw, npc_vehicles):
    dist = [1000] * len(npc_vehicles)
    count = -1
    if -45 < player_yaw < 45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 8:
                if -45 < np_vehicle['rotation']['yaw'] < 45:
                    if 0 < (np_vehicle['location']['x'] - player_x) < 35:
                        dist[count] = np_vehicle['location']['x'] - player_x
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] < 0.5:
                return True

    elif 45 < player_yaw < 135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 8:
                if 45 < np_vehicle['rotation']['yaw'] < 135:
                    if 0 < (np_vehicle['location']['y'] - player_y) < 35:
                        dist[count] = np_vehicle['location']['y'] - player_y
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] < 0.5:
                return True

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 8:
                if np_vehicle['rotation']['yaw'] > 135 or -180 <= np_vehicle['rotation']['yaw'] < -135:
                    if 0 < (player_x - np_vehicle['location']['x']) < 35:
                        dist[count] = player_x - np_vehicle['location']['x']
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] < 0.5:
                return True

    elif -135 < player_yaw < -45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 8:
                if -135 < np_vehicle['rotation']['yaw'] < -45:
                    if 0 < (player_y - np_vehicle['location']['y']) < 35:
                        dist[count] = player_y - np_vehicle['location']['y']
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] < 0.5:
                return True
    return False


def check_moving_vehicle(player_x, player_y, player_yaw, npc_vehicles):
    dist = [1000] * len(npc_vehicles)
    count = -1
    if -45 < player_yaw < 45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 8:
                if -45 < np_vehicle['rotation']['yaw'] < 45:
                    if 0 < (np_vehicle['location']['x'] - player_x) < 35:
                        dist[count] = np_vehicle['location']['x'] - player_x
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] > 0.5:
                return True

    elif 45 < player_yaw < 135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 8:
                if 45 < np_vehicle['rotation']['yaw'] < 135:
                    if 0 < (np_vehicle['location']['y'] - player_y) < 35:
                        dist[count] = np_vehicle['location']['y'] - player_y
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] > 0.5:
                return True

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 8:
                if np_vehicle['rotation']['yaw'] > 135 or -180 <= np_vehicle['rotation']['yaw'] < -135:
                    if 0 < (player_x - np_vehicle['location']['x']) < 35:
                        dist[count] = player_x - np_vehicle['location']['x']
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] > 0.5:
                return True

    elif -135 < player_yaw < -45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 8:
                if -135 < np_vehicle['rotation']['yaw'] < -45:
                    if 0 < (player_y - np_vehicle['location']['y']) < 35:
                        dist[count] = player_y - np_vehicle['location']['y']
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] > 0.5:
                return True
    return False


def check_stationary_vehicle_opposite_lane(player_x, player_y, player_yaw, npc_vehicles):
    dist = [1000] * len(npc_vehicles)
    count = -1
    if -45 < player_yaw < 45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 10:
                if np_vehicle['rotation']['yaw'] > 135 or -180 <= np_vehicle['rotation']['yaw'] < -135:
                    if 0 < (np_vehicle['location']['x'] - player_x) < 35:
                        dist[count] = np_vehicle['location']['x'] - player_x
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] < 0.5:
                return True

    elif 45 < player_yaw < 135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 10:
                if -135 < np_vehicle['rotation']['yaw'] < -45:
                    if 0 < (np_vehicle['location']['y'] - player_y) < 35:
                        dist[count] = np_vehicle['location']['y'] - player_y
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] < 0.5:
                return True

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 10:
                if -45 < np_vehicle['rotation']['yaw'] < 45:
                    if 0 < (player_x - np_vehicle['location']['x']) < 35:
                        dist[count] = player_x - np_vehicle['location']['x']
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] < 0.5:
                return True

    elif -135 < player_yaw < -45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 10:
                if 45 < np_vehicle['rotation']['yaw'] < 135:
                    if 0 < (player_y - np_vehicle['location']['y']) < 35:
                        dist[count] = player_y - np_vehicle['location']['y']
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] < 0.5:
                return True
    return False


def check_moving_vehicle_opposite_lane(player_x, player_y, player_yaw, npc_vehicles):
    dist = [1000] * len(npc_vehicles)
    count = -1
    if -45 < player_yaw < 45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 10:
                if np_vehicle['rotation']['yaw'] > 135 or -180 <= np_vehicle['rotation']['yaw'] < -135:
                    if 0 < (np_vehicle['location']['x'] - player_x) < 35:
                        dist[count] = np_vehicle['location']['x'] - player_x
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] > 0.5:
                return True

    elif 45 < player_yaw < 135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 10:
                if -135 < np_vehicle['rotation']['yaw'] < -45:
                    if 0 < (np_vehicle['location']['y'] - player_y) < 35:
                        dist[count] = np_vehicle['location']['y'] - player_y
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] > 0.5:
                return True

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 10:
                if -45 < np_vehicle['rotation']['yaw'] < 45:
                    if 0 < (player_x - np_vehicle['location']['x']) < 35:
                        dist[count] = player_x - np_vehicle['location']['x']
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] > 0.5:
                return True

    elif -135 < player_yaw < -45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 10:
                if 45 < np_vehicle['rotation']['yaw'] < 135:
                    if 0 < (player_y - np_vehicle['location']['y']) < 35:
                        dist[count] = player_y - np_vehicle['location']['y']
        closest_vehicle = dist.index(min(dist))
        if min(dist) is not 1000:
            if npc_vehicles[closest_vehicle]['forward_speed'] > 0.5:
                return True
    return False


def check_player_on_lane(other_lane, sidewalk_intersection):
    if other_lane < 0.1:
        if sidewalk_intersection < 0.1:
            return True
    return False


def check_player_in_opposite_lane(other_lane):
    if other_lane >= 0.1:
        return True
    return False


def check_player_on_sidewalk(sidewalk_intersection):
    if sidewalk_intersection >= 0.1:
        return True
    return False


def check_intersection_approaching(player_x, player_y, player_yaw, npc_traffic_light):
    dist = [1000] * len(npc_traffic_light)
    count = -1
    if -45 < player_yaw < 45:
        for np_traffic in npc_traffic_light:
            count = count + 1
            if abs(np_traffic['location']['y'] - player_y) < 20:
                if np_traffic['rotation']['yaw'] > 135 or -180 <= np_traffic['rotation']['yaw'] < -135:
                    if 0 < (np_traffic['location']['x'] - player_x) < 35:
                        dist[count] = np_traffic['location']['x'] - player_x
        closest_traffic_light = dist.index(min(dist))
        if min(dist) is not 1000:
            return True

    elif 45 < player_yaw < 135:
        for np_traffic in npc_traffic_light:
            count = count + 1
            if abs(np_traffic['location']['x'] - player_x) < 20:
                if -135 < np_traffic['rotation']['yaw'] < -45:
                    if 0 < (np_traffic['location']['y'] - player_y) < 35:
                        dist[count] = np_traffic['location']['y'] - player_y
        closest_traffic_light = dist.index(min(dist))
        if min(dist) is not 1000:
            return True

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_traffic in npc_traffic_light:
            count = count + 1
            if abs(np_traffic['location']['y'] - player_y) < 20:
                if -45 < np_traffic['rotation']['yaw'] < 45:
                    if 0 < (player_x - np_traffic['location']['x']) < 35:
                        dist[count] = player_x - np_traffic['location']['x']
        closest_traffic_light = dist.index(min(dist))
        if min(dist) is not 1000:
            return True

    elif -135 < player_yaw < -45:
        for np_traffic in npc_traffic_light:
            count = count + 1
            if abs(np_traffic['location']['x'] - player_x) < 20:
                if 45 < np_traffic['rotation']['yaw'] < 135:
                    if 0 < (player_y - np_traffic['location']['y']) < 35:
                        dist[count] = player_y - np_traffic['location']['y']
        closest_traffic_light = dist.index(min(dist))
        if min(dist) is not 1000:
            return True
    return False


def check_left_turn_approaching(player_x, player_y, player_yaw):
    """
        Left turn:
        345,330 to 380,330
        396, 50 to 396, 1
        50, -2 to 1, -2
        -2, 270 to -2 320

        Right turn:
        393, 289 to 392, 323
        347, 1 to 386, 2
        3, 64 to 2, 10
        50, 327 to  3, 321
    """

    if 340 < player_x < 385 and 329 < player_y < 331:
        if -10 < player_yaw < 80:
            return True

    elif 395 < player_x < 397 and 1 < player_y < 50:
        if -100 < player_yaw < 10:
            return True

    elif 1 < player_x < 50 and -2.5 < player_y < -1.5:
        if player_yaw > 170 or -180 < player_yaw < -80:
            return True

    elif -3 < player_x < -1 and 270 < player_y < 320:
        if 80 < player_yaw < 169:
            return True
    return False


def check_right_turn_approaching(player_x, player_y, player_yaw):
    if 392 < player_x < 394 and 289 < player_y < 323:
        if 10 < player_yaw < 100:
            return True

    elif 347 < player_x < 386 and 0 < player_y < 2:
        if -80 < player_yaw < 10:
            return True

    elif 2 < player_x < 4 and 10 < player_y < 64:
        if - 170 < player_yaw < - 80:
            return True

    elif 3 < player_x < 50 and 325 < player_y < 328:
        if player_yaw > 110 or -180 <= player_yaw <= -170:
            return True
    return False


# TODO Multiple pedestrian check
def check_pedestrian_approaching(player_x, player_y, player_yaw, npc_pedestrain):
    if -45 < player_yaw < 45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['y'] - player_y < 5:
                if 0 < (pedestrain['location']['x'] - player_x) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] <= player_y:
                        return True
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] >= player_y:
                        return True

    elif 45 < player_yaw < 135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['x'] - player_x < 1:
                if 0 < (pedestrain['location']['y'] - player_y) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] <= player_x:
                        return True
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] >= player_x):
                        return True

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['y'] - player_y < 1:
                if 0 < (player_x - pedestrain['location']['x']) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] <= player_y:
                        return True
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] >= player_y:
                        return True

    elif -135 < player_yaw < -45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['x'] - player_x < 5:
                if 0 < (player_y - pedestrain['location']['y']) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] <= player_x:
                        return True
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] >= player_x):
                        return True
    return False


# TODO Multiple pedestrian check
def check_pedestrian_departing(player_x, player_y, player_yaw, npc_pedestrain):
    if -45 < player_yaw < 45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['y'] - player_y < 5:
                if 0 < (pedestrain['location']['x'] - player_x) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] > player_y:
                        return True
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] < player_y:
                        return True

    elif 45 < player_yaw < 135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['x'] - player_x < 1:
                if 0 < (pedestrain['location']['y'] - player_y) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] > player_x:
                        return True
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] < player_x):
                        return True

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['y'] - player_y < 1:
                if 0 < (player_x - pedestrain['location']['x']) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] > player_y:
                        return True
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] < player_y:
                        return True

    elif -135 < player_yaw < -45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['x'] - player_x < 5:
                if 0 < (player_y - pedestrain['location']['y']) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] > player_x:
                        return True
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] < player_x):
                        return True
    return False


def check_stationary_object_ahead():
    return False


def check_pedestrian_on_sidewalk():
    return False


def dist_to_vehicle_same_lane(player_x, player_y, player_yaw, npc_vehicles):
    dist = [35] * len(npc_vehicles)
    count = -1
    if -45 < player_yaw < 45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 8:
                if -45 < np_vehicle['rotation']['yaw'] < 45:
                    if 0 < (np_vehicle['location']['x'] - player_x) < 35:
                        dist[count] = np_vehicle['location']['x'] - player_x
        closest_vehicle = dist.index(min(dist))
        return min(dist)

    elif 45 < player_yaw < 135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 8:
                if 45 < np_vehicle['rotation']['yaw'] < 135:
                    if 0 < (np_vehicle['location']['y'] - player_y) < 35:
                        dist[count] = np_vehicle['location']['y'] - player_y
        closest_vehicle = dist.index(min(dist))
        return min(dist)

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 8:
                if np_vehicle['rotation']['yaw'] > 135 or -180 <= np_vehicle['rotation']['yaw'] < -135:
                    if 0 < (player_x - np_vehicle['location']['x']) < 35:
                        dist[count] = player_x - np_vehicle['location']['x']
        closest_vehicle = dist.index(min(dist))
        return min(dist)

    elif -135 < player_yaw < -45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 8:
                if -135 < np_vehicle['rotation']['yaw'] < -45:
                    if 0 < (player_y - np_vehicle['location']['y']) < 35:
                        dist[count] = player_y - np_vehicle['location']['y']
        closest_vehicle = dist.index(min(dist))
        return min(dist)


def vel_of_vehicle_same_lane(player_x, player_y, player_yaw, npc_vehicles):
    dist = [100] * len(npc_vehicles)
    count = -1
    if -45 < player_yaw < 45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 8:
                if -45 < np_vehicle['rotation']['yaw'] < 45:
                    if 0 < (np_vehicle['location']['x'] - player_x) < 35:
                        dist[count] = np_vehicle['location']['x'] - player_x
        closest_vehicle = dist.index(min(dist))
        if min(dist) < 100:
            return npc_vehicles[closest_vehicle]['forward_speed']
        else:
            return -45

    elif 45 < player_yaw < 135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 8:
                if 45 < np_vehicle['rotation']['yaw'] < 135:
                    if 0 < (np_vehicle['location']['y'] - player_y) < 35:
                        dist[count] = np_vehicle['location']['y'] - player_y
        closest_vehicle = dist.index(min(dist))
        if min(dist) < 100:
            return npc_vehicles[closest_vehicle]['forward_speed']
        else:
            return -45

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 8:
                if np_vehicle['rotation']['yaw'] > 135 or -180 <= np_vehicle['rotation']['yaw'] < -135:
                    if 0 < (player_x - np_vehicle['location']['x']) < 35:
                        dist[count] = player_x - np_vehicle['location']['x']
        closest_vehicle = dist.index(min(dist))
        if min(dist) < 100:
            return npc_vehicles[closest_vehicle]['forward_speed']
        else:
            return -45

    elif -135 < player_yaw < -45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 8:
                if -135 < np_vehicle['rotation']['yaw'] < -45:
                    if 0 < (player_y - np_vehicle['location']['y']) < 35:
                        dist[count] = player_y - np_vehicle['location']['y']
        closest_vehicle = dist.index(min(dist))
        if min(dist) < 100:
            return npc_vehicles[closest_vehicle]['forward_speed']
        else:
            return -45


def dist_to_vehicle_opposite_lane(player_x, player_y, player_yaw, npc_vehicles):
    dist = [35] * len(npc_vehicles)
    count = -1
    if -45 < player_yaw < 45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 10:
                if np_vehicle['rotation']['yaw'] > 135 or -180 <= np_vehicle['rotation']['yaw'] < -135:
                    if 0 < (np_vehicle['location']['x'] - player_x) < 35:
                        dist[count] = np_vehicle['location']['x'] - player_x
        closest_vehicle = dist.index(min(dist))
        return min(dist)

    elif 45 < player_yaw < 135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 10:
                if -135 < np_vehicle['rotation']['yaw'] < -45:
                    if 0 < (np_vehicle['location']['y'] - player_y) < 35:
                        dist[count] = np_vehicle['location']['y'] - player_y
        closest_vehicle = dist.index(min(dist))
        return min(dist)

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 10:
                if -45 < np_vehicle['rotation']['yaw'] < 45:
                    if 0 < (player_x - np_vehicle['location']['x']) < 35:
                        dist[count] = player_x - np_vehicle['location']['x']
        closest_vehicle = dist.index(min(dist))
        return min(dist)

    elif -135 < player_yaw < -45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 10:
                if 45 < np_vehicle['rotation']['yaw'] < 135:
                    if 0 < (player_y - np_vehicle['location']['y']) < 35:
                        dist[count] = player_y - np_vehicle['location']['y']
        closest_vehicle = dist.index(min(dist))
        return min(dist)


def vel_of_vehicle_opposite_lane(player_x, player_y, player_yaw, npc_vehicles):
    dist = [100] * len(npc_vehicles)
    count = -1
    if -45 < player_yaw < 45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 10:
                if np_vehicle['rotation']['yaw'] > 135 or -180 <= np_vehicle['rotation']['yaw'] < -135:
                    if 0 < (np_vehicle['location']['x'] - player_x) < 35:
                        dist[count] = np_vehicle['location']['x'] - player_x
        closest_vehicle = dist.index(min(dist))
        if min(dist) < 100:
            return npc_vehicles[closest_vehicle]['forward_speed']
        else:
            return -45

    elif 45 < player_yaw < 135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 10:
                if -135 < np_vehicle['rotation']['yaw'] < -45:
                    if 0 < (np_vehicle['location']['y'] - player_y) < 35:
                        dist[count] = np_vehicle['location']['y'] - player_y
        closest_vehicle = dist.index(min(dist))
        if min(dist) < 100:
            return npc_vehicles[closest_vehicle]['forward_speed']
        else:
            return -45

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['y'] - player_y) < 10:
                if -45 < np_vehicle['rotation']['yaw'] < 45:
                    if 0 < (player_x - np_vehicle['location']['x']) < 35:
                        dist[count] = player_x - np_vehicle['location']['x']
        closest_vehicle = dist.index(min(dist))
        if min(dist) < 100:
            return npc_vehicles[closest_vehicle]['forward_speed']
        else:
            return -45

    elif -135 < player_yaw < -45:
        for np_vehicle in npc_vehicles:
            count = count + 1
            if abs(np_vehicle['location']['x'] - player_x) < 10:
                if 45 < np_vehicle['rotation']['yaw'] < 135:
                    if 0 < (player_y - np_vehicle['location']['y']) < 35:
                        dist[count] = player_y - np_vehicle['location']['y']
        closest_vehicle = dist.index(min(dist))
        if min(dist) < 100:
            return npc_vehicles[closest_vehicle]['forward_speed']
        else:
            return -45

def distance_to_approaching_intersection(player_x, player_y, player_yaw, npc_traffic_light):
    dist = [35] * len(npc_traffic_light)
    count = -1
    if -45 < player_yaw < 45:
        for np_traffic in npc_traffic_light:
            count = count + 1
            if abs(np_traffic['location']['y'] - player_y) < 20:
                # if np_traffic['rotation']['yaw'] > 135 or -180 <= np_traffic['rotation']['yaw'] < -135:
                if 0 < (np_traffic['location']['x'] - player_x) < 35:
                    dist[count] = np_traffic['location']['x'] - player_x
        closest_traffic_light = dist.index(min(dist))
        return min(dist)

    elif 45 < player_yaw < 135:
        for np_traffic in npc_traffic_light:
            count = count + 1
            if abs(np_traffic['location']['x'] - player_x) < 20:
                #if -135 < np_traffic['rotation']['yaw'] < -45:
                if 0 < (np_traffic['location']['y'] - player_y) < 35:
                    dist[count] = np_traffic['location']['y'] - player_y
        closest_traffic_light = dist.index(min(dist))
        return min(dist)

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for np_traffic in npc_traffic_light:
            count = count + 1
            if abs(np_traffic['location']['y'] - player_y) < 20:
                #if -45 < np_traffic['rotation']['yaw'] < 45:
                if 0 < (player_x - np_traffic['location']['x']) < 35:
                    dist[count] = player_x - np_traffic['location']['x']
        closest_traffic_light = dist.index(min(dist))
        return min(dist)

    elif -135 < player_yaw < -45:
        for np_traffic in npc_traffic_light:
            count = count + 1
            if abs(np_traffic['location']['x'] - player_x) < 20:
                #if 45 < np_traffic['rotation']['yaw'] < 135:
                if 0 < (player_y - np_traffic['location']['y']) < 35:
                    dist[count] = player_y - np_traffic['location']['y']
        closest_traffic_light = dist.index(min(dist))
        return min(dist)


def distance_to_approaching_left_turn(player_x, player_y, player_yaw):
    """
            Left turn:
            345,330 to 380,330
            396, 50 to 396, 1
            50, -2 to 1, -2
            -2, 270 to -2 320

            Right turn:
            393, 289 to 392, 323
            347, 1 to 386, 2
            3, 64 to 2, 10
            50, 327 to  3, 321
        """

    if 340 < player_x < 385 and 327 < player_y < 333:
        if -10 < player_yaw < 80:
            return 385 - player_x

    elif 393 < player_x < 400 and 1 < player_y < 50:
        if -100 < player_yaw < 10:
            return player_y

    elif 1 < player_x < 50 and -5 < player_y < 5:
        if player_yaw > 170 or -180 < player_yaw < -80:
            return player_x

    elif -5 < player_x < 0 and 270 < player_y < 320:
        if 80 < player_yaw < 169:
            return 321 - player_y
    return -50


def distance_to_approaching_right_turn(player_x, player_y, player_yaw):
    if 390 < player_x < 400 and 289 < player_y < 323:
        if 10 < player_yaw < 100:
            return 331 - player_y

    elif 347 < player_x < 386 and -2 < player_y < 4:
        if -80 < player_yaw < 10:
            return 386 - player_x

    elif 0 < player_x < 6 and 10 < player_y < 64:
        if - 170 < player_yaw < - 80:
            return player_y - 10

    elif 3 < player_x < 50 and 323 < player_y < 330 :
        if player_yaw > 110 or -180 <= player_yaw <= -170:
            return player_x - 3
    return -50


def longitudinal_distance_to_approaching_pedestrian(player_x, player_y, player_yaw, npc_pedestrain):
    if -45 < player_yaw < 45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['y'] - player_y < 5:
                if 0 < (pedestrain['location']['x'] - player_x) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] <= player_y:
                        return pedestrain['location']['x'] - player_x
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] >= player_y:
                        return pedestrain['location']['x'] - player_x

    elif 45 < player_yaw < 135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['x'] - player_x < 1:
                if 0 < (pedestrain['location']['y'] - player_y) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] <= player_x:
                        return pedestrain['location']['y'] - player_y
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] >= player_x):
                        return pedestrain['location']['y'] - player_y

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['y'] - player_y < 1:
                if 0 < (player_x - pedestrain['location']['x']) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] <= player_y:
                        return pedestrain['location']['x'] - player_x
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] >= player_y:
                        return pedestrain['location']['x'] - player_x

    elif -135 < player_yaw < -45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['x'] - player_x < 5:
                if 0 < (player_y - pedestrain['location']['y']) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] <= player_x:
                        return pedestrain['location']['y'] - player_y
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] >= player_x):
                        return pedestrain['location']['y'] - player_y
    return -15


def lateral_distance_to_approaching_pedestrian(player_x, player_y, player_yaw, npc_pedestrain):
    if -45 < player_yaw < 45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['y'] - player_y < 5:
                if 0 < (pedestrain['location']['x'] - player_x) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] <= player_y:
                        return pedestrain['location']['y'] - player_y
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] >= player_y:
                        return pedestrain['location']['y'] - player_y

    elif 45 < player_yaw < 135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['x'] - player_x < 1:
                if 0 < (pedestrain['location']['y'] - player_y) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] <= player_x:
                        return pedestrain['location']['x'] - player_x
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] >= player_x):
                        return pedestrain['location']['x'] - player_x

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['y'] - player_y < 1:
                if 0 < (player_x - pedestrain['location']['x']) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] <= player_y:
                        return pedestrain['location']['y'] - player_y
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] >= player_y:
                        return pedestrain['location']['y'] - player_y

    elif -135 < player_yaw < -45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['x'] - player_x < 5:
                if 0 < (player_y - pedestrain['location']['y']) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] <= player_x:
                        return pedestrain['location']['x'] - player_x
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] >= player_x):
                        return pedestrain['location']['x'] - player_x
    return -8


def longitudinal_distance_to_departing_pedestrian(player_x, player_y, player_yaw, npc_pedestrain):
    if -45 < player_yaw < 45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['y'] - player_y < 5:
                if 0 < (pedestrain['location']['x'] - player_x) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] > player_y:
                        return pedestrain['location']['x'] - player_x
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] < player_y:
                        return pedestrain['location']['x'] - player_x

    elif 45 < player_yaw < 135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['x'] - player_x < 1:
                if 0 < (pedestrain['location']['y'] - player_y) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] > player_x:
                        return pedestrain['location']['y'] - player_y
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] < player_x):
                        return pedestrain['location']['y'] - player_y

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['y'] - player_y < 1:
                if 0 < (player_x - pedestrain['location']['x']) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] > player_y:
                        return pedestrain['location']['x'] - player_x
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] < player_y:
                        return pedestrain['location']['x'] - player_x

    elif -135 < player_yaw < -45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['x'] - player_x < 5:
                if 0 < (player_y - pedestrain['location']['y']) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] > player_x:
                        return pedestrain['location']['y'] - player_y
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] < player_x):
                        return pedestrain['location']['y'] - player_y
    return -15


def lateral_distance_to_departing_pedestrian(player_x, player_y, player_yaw, npc_pedestrain):
    if -45 < player_yaw < 45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['y'] - player_y < 5:
                if 0 < (pedestrain['location']['x'] - player_x) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] > player_y:
                        return pedestrain['location']['y'] - player_y
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] < player_y:
                        return pedestrain['location']['y'] - player_y

    elif 45 < player_yaw < 135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['x'] - player_x < 1:
                if 0 < (pedestrain['location']['y'] - player_y) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] > player_x:
                        return pedestrain['location']['x'] - player_x
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] < player_x):
                        return pedestrain['location']['x'] - player_x

    elif player_yaw > 135 or -180 <= player_yaw < -135:
        for pedestrain in npc_pedestrain:
            if -5 < pedestrain['location']['y'] - player_y < 1:
                if 0 < (player_x - pedestrain['location']['x']) < 15:
                    if 45 < pedestrain['rotation']['yaw'] < 135 and pedestrain['location']['y'] > player_y:
                        return pedestrain['location']['y'] - player_y
                    elif -135 < pedestrain['rotation']['yaw'] < -45 and pedestrain['location']['y'] < player_y:
                        return pedestrain['location']['y'] - player_y

    elif -135 < player_yaw < -45:
        for pedestrain in npc_pedestrain:
            if -1 < pedestrain['location']['x'] - player_x < 5:
                if 0 < (player_y - pedestrain['location']['y']) < 15:
                    if -45 < pedestrain['rotation']['yaw'] < 45 and pedestrain['location']['x'] > player_x:
                        return pedestrain['location']['x'] - player_x
                    elif (pedestrain['rotation']['yaw'] > 135 or -180 <= pedestrain['rotation']['yaw'] < -135) and \
                            (pedestrain['location']['x'] < player_x):
                        return pedestrain['location']['x'] - player_x
    return -8



def main():

    """
    action abstractions:
        0 do nothing
        1 speed up
        2 slow down
        3 stop
        4 turn left
        5 turn right
        6 cut out
        7 cut in

    visual abstractions:
        0 distance vehicle ahead in same lane
        1 velocity of moving vehicle ahead in same lane
        2 distance to vehicle ahead in opposite lane
        3 velocity of vehicle ahead in opposite lane
        4 percentage player in opposite lane
        5 percentage player on sidewalk
        6 distance to approaching intersection
        7 distance to approaching left turn
        8 distance to approaching right turn
        9 longitudinal distance to pedestrian approaching
        10 lateral distance to pedestrian approaching
        11 longitudinal distance to pedestrian departing
        12 lateral distance to pedestrian departing


        11 stationary object ahead
        11 pedestrian on sidewalk

        yaw 0: x increases
        yaw -180: x decreases
        yaw 90: y increases
        yaw -90: y decreases
    """

    DIRNAME = '/media/ashish/C846824B46823A68/DataSets/NEW_CARLA_Steering_dataset/Data_Set/'
    print(DIRNAME)
    for i in sorted(glob.glob(DIRNAME + '*')):
        for j in sorted(glob.glob(i + '/*.h5')):
            print("Processing file:", j)
            h5file = h5py.File(j, "r+")
            targets = h5file['targets']
            #targets.resize(targets.shape[1] + 13, axis=1)
            # image = h5file['Camera/RGB_front']
            json_folder = j[:-3]
            print(h5file['targets'].shape[0])
            control = [0, 0, 0]
            control_t1 = [0, 0, 0]
            other_lane_t1 = 0
            action_annotation = 0b0000000000000000
            action_annotation_t1 = 0b0000000000000000

            for index in range(targets.shape[0] - 10):
                measurements = json.load(open(json_folder + '/measurements{}.json'.format(index)))
                control = [targets[index, 0], targets[index, 1], targets[index, 2]]  # steering, throttle, brake
                speed = targets[index, 10]
                other_lane = targets[index, 14]

                # action_annotation = 0b0000000000000000
                # if check_speed_up(control, control_t1):
                #     action_annotation = action_annotation | 0b0000000000000010
                # elif check_slow_down(control, control_t1):
                #     action_annotation = action_annotation | 0b0000000000000100
                # elif check_stop(control):
                #     action_annotation = action_annotation | 0b0000000000001000
                # elif check_do_nothing(control, speed):
                #     action_annotation = action_annotation | 0b0000000000000001
                # if check_left(control):
                #     action_annotation = action_annotation | 0b0000000000010000
                # elif check_right(control):
                #     action_annotation = action_annotation | 0b0000000000100000
                # if check_cut_out(control, other_lane, other_lane_t1):
                #     action_annotation = action_annotation | 0b0000000001000000
                # elif check_cut_in(control, other_lane, other_lane_t1):
                #     action_annotation = action_annotation | 0b0000000010000000
                #
                # if action_annotation == 0b0000000000000000:
                #     action_annotation = action_annotation_t1
                #
                # targets[index, 25] = action_annotation

                player_pos_x = measurements['PlayerMeasurements']['location']['x']
                player_pos_y = measurements['PlayerMeasurements']['location']['y']
                player_yaw = measurements['PlayerMeasurements']['rotation']['yaw']
                npc_measurements = measurements['NpcMeasurements']
                sidewalk_intersection = targets[index, 15]
                planner_command = targets[index, 24]

                npc_vehicles = []
                npc_pedestrain = []
                npc_traffic_light = []
                for k in range(npc_measurements.__len__()):
                    if npc_measurements[k]['type'] == 'vehicle':
                        npc_vehicles.append(npc_measurements[k])
                    elif npc_measurements[k]['type'] == 'pedestrian':
                        npc_pedestrain.append(npc_measurements[k])
                    elif npc_measurements[k]['type'] == 'traffic_light':
                        npc_traffic_light.append(npc_measurements[k])

                # visual_annotation = 0b0000000000000000
                #
                # if check_stationary_vehicle(player_pos_x, player_pos_y, player_yaw, npc_vehicles):
                #     visual_annotation = visual_annotation | 0b0000000000000001
                # if check_moving_vehicle(player_pos_x, player_pos_y, player_yaw, npc_vehicles):
                #     visual_annotation = visual_annotation | 0b0000000000000010
                # if check_stationary_vehicle_opposite_lane(player_pos_x, player_pos_y, player_yaw, npc_vehicles):
                #     visual_annotation = visual_annotation | 0b0000000000000100
                # if check_moving_vehicle_opposite_lane(player_pos_x, player_pos_y, player_yaw, npc_vehicles):
                #     visual_annotation = visual_annotation | 0b0000000000001000
                # if check_player_on_lane(other_lane, sidewalk_intersection):
                #     visual_annotation = visual_annotation | 0b0000000000010000
                # if check_player_in_opposite_lane(other_lane):
                #     visual_annotation = visual_annotation | 0b0000000000100000
                # if check_player_on_sidewalk(sidewalk_intersection):
                #     visual_annotation = visual_annotation | 0b0000000001000000
                # if check_intersection_approaching(player_pos_x, player_pos_y, player_yaw, npc_traffic_light):
                #     visual_annotation = visual_annotation | 0b0000000010000000
                # if check_left_turn_approaching(player_pos_x, player_pos_y, player_yaw):
                #     visual_annotation = visual_annotation | 0b0000000100000000
                # if check_right_turn_approaching(player_pos_x, player_pos_y, player_yaw):
                #     visual_annotation = visual_annotation | 0b0000001000000000
                # if check_pedestrian_approaching(player_pos_x, player_pos_y, player_yaw, npc_pedestrain):
                #     visual_annotation = visual_annotation | 0b0000010000000000
                # if check_pedestrian_departing(player_pos_x, player_pos_y, player_yaw, npc_pedestrain):
                #     visual_annotation = visual_annotation | 0b0000100000000000
                # #     visual_annotation = visual_annotation | 0b0001000000000000
                # # if check_stationary_object_ahead():
                # #     visual_annotation = visual_annotation | 0b0010000000000000
                # # if check_pedestrian_on_sidewalk():
                # #     visual_annotation = visual_annotation | 0b0100000000000000
                #
                # targets[index, 26] = visual_annotation

                targets[index, 34] = dist_to_vehicle_same_lane(player_pos_x, player_pos_y,
                                                                      player_yaw, npc_vehicles)/35

                targets[index, 35] = vel_of_vehicle_same_lane(player_pos_x, player_pos_y, player_yaw, npc_vehicles)/45

                targets[index, 36] = dist_to_vehicle_opposite_lane(player_pos_x, player_pos_y,
                                                                   player_yaw, npc_vehicles)/35
                targets[index, 37] = vel_of_vehicle_opposite_lane(player_pos_x, player_pos_y,
                                                                  player_yaw, npc_vehicles)/45
                targets[index, 38] = targets[index, 14]

                targets[index, 39] = targets[index, 15]

                targets[index, 40] = distance_to_approaching_intersection (player_pos_x, player_pos_y,
                                                                           player_yaw, npc_traffic_light)/35

                targets[index, 41] = distance_to_approaching_left_turn(player_pos_x, player_pos_y, player_yaw)/50

                targets[index, 42] = distance_to_approaching_right_turn(player_pos_x, player_pos_y, player_yaw)/50

                targets[index, 43] = longitudinal_distance_to_approaching_pedestrian(player_pos_x, player_pos_y,
                                                                                     player_yaw, npc_pedestrain)/15

                targets[index, 44] = lateral_distance_to_approaching_pedestrian(player_pos_x, player_pos_y,
                                                                                     player_yaw, npc_pedestrain)/8

                targets[index, 45] = longitudinal_distance_to_departing_pedestrian(player_pos_x, player_pos_y,
                                                                                     player_yaw, npc_pedestrain)/15

                targets[index, 46] = lateral_distance_to_departing_pedestrian(player_pos_x, player_pos_y,
                                                                                     player_yaw, npc_pedestrain)/8


                control_t1 = control
                other_lane_t1 = other_lane
                action_annotation_t1 = action_annotation
main()
