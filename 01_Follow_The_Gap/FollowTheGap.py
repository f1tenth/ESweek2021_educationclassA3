import time
import gym
import numpy as np
import concurrent.futures
import os
import sys
import yaml
from argparse import Namespace

# Get ./src/ folder & add it to path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# Import your drivers here

from drivers import GapFollower

# Choose your drivers for each of the cars you have on the track here.
drivers = [GapFollower()]

# Choose your racetrack here
RACETRACK = 'Spielberg'


class GymRunner(object):

    def __init__(self, racetrack, drivers):
        self.racetrack = racetrack
        self.drivers = drivers

    def run(self):

        # load map and the specific information for the map
        with open('config_Spielberg_obs_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)

        # Create the F1TENTH GYM environment
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)

        # Specify starting positions of each agent
        driver_count = len(drivers)
        if driver_count == 1:
            poses = np.array([[0.8007017, -0.2753365, 4.1421595]])
        elif driver_count == 2:
            poses = np.array([
                [0.8007017, -0.2753365, 4.1421595],
                [0.8162458, 1.1614572, 4.1446321],
            ])
        else:
            raise ValueError("Max 2 drivers are allowed")

        # Initially parametrize the environment
        obs, step_reward, done, info = env.reset(poses=poses)
        env.render()
        laptime = 0.0
        start = time.time()

        # Start the simulation
        while not done:
            actions = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, driver in enumerate(drivers):
                    futures.append(executor.submit(driver.process_lidar, obs['scans'][i]))
            for future in futures:
                # Get the actions based on the Follow the Gap Algorithm
                speed, steer = future.result()
                actions.append([steer, speed])
            actions = np.array(actions)

            # Send the actions from the Follow the Gap to the simulation environment
            obs, step_reward, done, info = env.step(actions)
            laptime += step_reward
            env.render(mode='human')

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    runner = GymRunner(RACETRACK, drivers)
    runner.run()
