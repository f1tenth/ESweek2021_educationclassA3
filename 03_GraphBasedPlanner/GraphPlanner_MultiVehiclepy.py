import sys
import os

# -- Limit number of OPENBLAS library threads --
# On linux based operation systems, we observed a occupation of all cores by the underlying openblas library. Often,
# this slowed down other processes, as well as the planner itself. Therefore, it is recommended to set the number of
# threads to one. Note: this import must happen before the import of any openblas based package (e.g. numpy)
os.environ['OPENBLAS_NUM_THREADS'] = str(1)

import numpy as np
import datetime
import json
import time
import configparser
import yaml
import gym
from argparse import Namespace
import graph_ltpl
from numba import njit
import math

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

# @njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


@njit(fastmath=False, cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

    def load_waypoints(self, conf):
        # load waypoints
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle

class Controllers:
    """
    This is the PurePursuit ALgorithm that is traccking the desired path. In this case we are following the curvature
    optimal raceline.
    """

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.max_reacquire = 20.
        self.vehicle_control_e_f = 0            # Control error
        self.vehicle_control_error3 = 0

    def _get_current_waypoint(self, lookahead_distance, position, traj_set, sel_action):
        # Check which trajectory set is available and select one
        for sel_action in ["right", "left", "straight", "follow"]:  # try to force 'right', else try next in list
            if sel_action in traj_set.keys():
                break

        # Extract Trajectory informtion from the current set: X-Position, Y-Position, Velocity
        path_x = traj_set[sel_action][0][:,1]
        path_y = traj_set[sel_action][0][:, 2]
        velocity = traj_set[sel_action][0][:, 5]
        # Create waypoints based on the current path
        wpts = np.vstack((np.array(path_x), np.array(path_y))).T

        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        #print ('nearest distance: ', nearest_dist)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts,
                                                                                    i + t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = velocity[i2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], velocity[i])
        else:
            return None

    def PurePursuit(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain, traj_set, sel_action):
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(lookahead_distance, position, traj_set,sel_action)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle

    def calc_theta_and_ef(self, vehicle_state, waypoints, goal_heading, goal_velocity):
        """
        calc theta and ef
        Theta is the heading of the car, this heading must be minimized
        ef = crosstrack error/The distance from the optimal path/ lateral distance in frenet frame (front wheel)
        """

        ############# Calculate closest point to the front axle based on minimum distance calculation ################
        # Calculate Position of the front axle of the vehicle based on current position
        fx = vehicle_state[0] + self.wheelbase * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase * math.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])

        # Find target index for the correct waypoint by finding the index with the lowest distance value/hypothenuses
        #wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point_front, nearest_dist, t, target_index = nearest_point_on_trajectory(position_front_axle, waypoints)

        # Calculate the Distances from the front axle to all the waypoints
        distance_nearest_point_x = fx - nearest_point_front[0]
        distance_nearest_point_y = fy - nearest_point_front[1]
        vec_dist_nearest_point = np.array([distance_nearest_point_x, distance_nearest_point_y])

        ###################  Calculate the current Cross-Track Error ef in [m]   ################
        # Project crosstrack error onto front axle vector
        front_axle_vec_rot_90 = np.array([[math.cos(vehicle_state[2] - math.pi / 2.0)],
                                          [math.sin(vehicle_state[2] - math.pi / 2.0)]])

        # vec_target_2_front = np.array([dx[target_index], dy[target_index]])

        # Caculate the cross-track error ef by
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

        #############  Calculate the heading error theta_e  normalized to an angle to [-pi, pi]     ##########
        # Extract heading on the raceline
        # BE CAREFUL: If your raceline is based on a different coordinate system you need to -+ pi/2 = 90 degrees
        theta_raceline = goal_heading[target_index] + np.pi/2

        # Calculate the heading error by taking the difference between current and goal + Normalize the angles
        theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

        # Calculate the target Veloctiy for the desired state
        planned_veloctiy = goal_velocity[target_index]

        return theta_e, ef, target_index, planned_veloctiy

    def StanleyController(self, pose_x, pose_y, pose_theta, current_velocity, vgain, traj_set, sel_action):
        """
        Front Wheel Feedback Controller to track the path
        Based on the heading error theta_e and the crosstrack error ef we calculate the steering angle
        Returns the optimal steering angle delta is P-Controller with the proportional gain k
        """

        # Check which trajectory set is available and select one
        for sel_action in ["right", "left", "straight", "follow"]:  # try to force 'right', else try next in list
            if sel_action in traj_set.keys():
                break

        # Extract Trajectory informtion from the current set: X-Position, Y-Position, Velocity
        path_x = traj_set[sel_action][0][:, 1]
        path_y = traj_set[sel_action][0][:, 2]
        heading = traj_set[sel_action][0][:, 3]
        velocity = traj_set[sel_action][0][:, 5]
        # Create waypoints based on the current path
        wpts = np.vstack((np.array(path_x), np.array(path_y))).T

        kp = 8.63010407                     # Proportional gain for path control
        kd = 1.45                           # Differential gain
        ki = 0.6                            # Integral gain

        vehicle_state = np.array([pose_x, pose_y, pose_theta, current_velocity])
        theta_e, ef, target_index, goal_velocity = self.calc_theta_and_ef(vehicle_state, wpts, heading, velocity)

        # PID Stanly: This is Stanly with Integral (I) and Differential (D) calculations
        # Caculate steering angle based on the cross track error to the front axle in [rad]
        error1 = (kp * ef[0])
        error2 = (kd * (ef[0] - self.vehicle_control_e_f) / 0.01)
        error3 = self.vehicle_control_error3 + (ki * ef[0] * 0.01)
        error = error1 + error2 + error3
        cte_front = math.atan2(error, vehicle_state[3])
        self.vehicle_control_e_f = ef
        self.vehicle_control_error3 = error3

        # Classical Stanley: This is Stanly only with Proportional (P) calculations
        # Caculate steering angle based on the cross track error to the front axle in [rad]
        # cte_front = math.atan2(kp * ef, vehicle_state[3])

        # Calculate final steering angle/ control input in [rad]: Steering Angle based on error + heading error
        steering_angle = cte_front + theta_e

        # Calculate final speed control input in [m/s]:
        # speed_diff = k_veloctiy * (goal_veloctiy-velocity)
        speed = goal_velocity * vgain

        return steering_angle, speed


class GraphBasedPlanner:

    def __init__(self, conf):
        self.conf = conf
        self.init_flag = 0
        self.ltpl_obj = 0
        self.traj_set = 0
        self.zone_example = 0
        self.obj_list_dummy = 0

    def initialize_planner(self,conf):
        # ----------------------------------------------------------------------------------------------------------------------
        # IMPORT (should not change) -------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------

        # top level path (module directory)
        toppath = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(toppath)

        track_param = configparser.ConfigParser()
        if not track_param.read(toppath + "/params/driving_task.ini"):
            raise ValueError('Specified online parameter config file does not exist or is empty!')

        track_specifier = json.loads(track_param.get('DRIVING_TASK', 'track'))

        # define all relevant paths
        path_dict = {'globtraj_input_path': toppath + "/inputs/traj_ltpl_cl/traj_ltpl_cl_" + track_specifier + ".csv",
                     'graph_store_path': toppath + "/inputs/stored_graph.pckl",
                     'ltpl_offline_param_path': toppath + "/params/ltpl_config_offline.ini",
                     'ltpl_online_param_path': toppath + "/params/ltpl_config_online.ini",
                     'log_path': toppath + "/logs/graph_ltpl/",
                     'graph_log_id': datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                     }

        # ----------------------------------------------------------------------------------------------------------------------
        # INITIALIZATION AND OFFLINE GRAPH CALCULATION PART --------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------

        # intialize graph_ltpl-class
        ltpl_obj = graph_ltpl.Graph_LTPL.Graph_LTPL(path_dict=path_dict,
                                                    visual_mode=True,
                                                    log_to_file=True)

        # calculate offline graph
        ltpl_obj.graph_init()

        # set start pose based on first point in provided reference-line
        #refline = graph_ltpl.imp_global_traj.src.import_globtraj_csv. \
        #    import_globtraj_csv(import_path=path_dict['globtraj_input_path'])[0]
        #pos_est = refline[0, :]
        #heading_est = np.arctan2(np.diff(refline[0:2, 1]), np.diff(refline[0:2, 0])) - np.pi / 2

        pos_est = np.array([self.conf.sx, self.conf.sy])
        heading_est = np.array([round(self.conf.stheta - math.pi/2, 7)])

        # set start pos
        ltpl_obj.set_startpos(pos_est=pos_est,
                              heading_est=heading_est)


        # -- INIT DUMMY OBJECT LIST -----------------------------------------------------------------------
        #        * dynamic: TRUE = moving object, FALSE = static object
        #        * vel_scale: scale of velocity relativ to own vehicle
        #        * s0 = Starting s-position along the raceline of the object (dynamic only)
        obj_list_dummy = graph_ltpl.testing_tools.src.objectlist_dummy.ObjectlistDummy(dynamic=False,
                                                                                       vel_scale=0.5,
                                                                                       s0=50.0)

        # -- INIT SAMPLE ZONE -----------------------------------------------------------------------
        # (NOTE: only valid with the default track and configuration!)
        # INFO: Zones can be used to temporarily block certain regions (e.g. pit lane, accident region, dirty track, ....).
        #       Each zone is specified in a as a dict entry, where the key is the zone ID and the value is a list with the cells
        #        * blocked layer numbers (in the graph) - pairwise with blocked node numbers
        #        * blocked node numbers (in the graph) - pairwise with blocked layer numbers
        #        * numpy array holding coordinates of left bound of region (columns x and y)
        #        * numpy array holding coordinates of right bound of region (columns x and y)

        #zone_example = { 'sample_zone': [[64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66],
        #                    [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
        #                    np.array([[-20.54, 227.56], [23.80, 186.64]]),
        #                    np.array([[-23.80, 224.06], [20.17, 183.60]])]}

        zone_example ={}

        traj_set = {'straight': None}

        return ltpl_obj,traj_set, zone_example, obj_list_dummy


    def plan(self, pose_x, pose_y, pose_theta, velocity, obstacle1):

        # -- INITIALIZE PLANNER ----------------------------------------------------------------------------------------
        if self.init_flag == 0:
            self.ltpl_obj,self.traj_set, self.zone_example, self.obj_list_dummy = self.initialize_planner(self.conf)
            self.init_flag =1

        # --------------------------------------------------------------------------------------------------------------
        # ONLINE LOOP --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        tic = time.time()
        # -- SELECT ONE OF THE PROVIDED TRAJECTORIES -------------------------------------------------------------------
        # (here: brute-force, replace by sophisticated behavior planner)
        for sel_action in ["right", "left", "straight", "follow"]:  # try to force 'right', else try next in list
            if sel_action in self.traj_set.keys():
                break

        # -- OBJECT LIST: GET INFORMATION ABOUT STATIC OR DYNAMIC OBSTACLES --------------------------------------------
        # Get simple object list
        obj_list = self.obj_list_dummy.get_objectlist()
        obj_list[0]['X'] = obstacle1[0]
        obj_list[0]['Y'] = obstacle1[1]
        obj_list[0]['theta'] = obstacle1[2] - np.pi/2
        obj_list[0]['v'] = obstacle1[3]

        # -- CALCULATE PATHS FOR NEXT TIMESTAMP ------------------------------------------------------------------------
        self.ltpl_obj.calc_paths(prev_action_id=sel_action,
                                 object_list=obj_list,
                                 blocked_zones=self.zone_example)

        # -- GET POSITION AND VELOCITY ESTIMATE OF EGO-VEHICLE ---------------------------------------------------------
        # vehicle_state = np.array([pose_x, pose_y, pose_theta, velocity])
        pos_est = np.array([pose_x, pose_y])
        vel_est = velocity
        tic = time.time()

        # -- CALCULATE VELOCITY PROFILE AND RETRIEVE TRAJECTORIES ------------------------------------------------------
        self.traj_set = self.ltpl_obj.calc_vel_profile(pos_est=pos_est,
                                                       vel_est=vel_est)[0]

        # -- LIVE PLOT (if activated - not recommended for performance use) --------------------------------------------
        self.ltpl_obj.visual()

        # -- LOGGING ---------------------------------------------------------------------------------------------------
        self.ltpl_obj.log()

        return self.traj_set, sel_action

    def control(self, pose_x, pose_y, pose_theta, current_velocity, traj_set, sel_action):

        # -- SEND TRAJECTORIES TO CONTROLLER -------------------------------------------------------------------------------
        # select a trajectory from the set and send it to the controller here

        # Fast Setup: lookehead: 1.05, vgain: 0.92
        #speed, steering_angle = controller.PurePursuit(pose_x, pose_y, pose_theta, 1.05, 0.80, traj_set, sel_action)
        steering_angle, speed = controller.StanleyController(pose_x, pose_y, pose_theta, current_velocity, 0.75, traj_set, sel_action)

        #print('Planned Speed:', speed, 'Current Speed:', velocity)

        return speed, steering_angle


if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.50}
    with open('config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta],[conf.sx2, conf.sy2, conf.stheta2]]))
    env.render()
    planner = GraphBasedPlanner(conf)
    controller = Controllers(conf, 0.17145 + 0.15875)
    planner2 = PurePursuitPlanner(conf, 0.17145 + 0.15875)

    laptime = 0.0
    control_count = 15
    start = time.time()

    while not done:

        if control_count == 15:
            # Get and gather information about the obstacles
            obstacle1 = [obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1],obs['linear_vels_x'][1]]
            # Run graph based planner. Receive set of trajectories and final selection
            traj_set, sel_action = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],obs['linear_vels_x'][0],obstacle1)
            # Reset Planner counter to zero
            control_count = 0

        speed, steer = planner.control(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],obs['linear_vels_x'][0],traj_set, sel_action)
        speed2, steer2 = planner2.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], work['tlad'],work['vgain'])
        control_count = control_count + 1

        obs, step_reward, done, info = env.step(np.array([[steer, speed],[steer2, speed2]]))
        laptime += step_reward
        env.render(mode='human_fast')
    print("Racetrack")
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)