# ESweek2021_educationclassA3
This is the repository that includes the code material for the ESweek 2021 for the Education Class Lecture A3 "Learn to Drive (and Race!) Autonomous Vehicles"

# Environment Installation guide

1. Create an Virtual Environment with Python 3.8
2. Install the required python packages with the following command:

```bash
pip3 install -r requirements.txt
```
3. Install the F1TENTH gym environment while in the root folder of this repository by running the following command:
```bash
$ pip3 install -e gym/
```
4. For more information about the F1TENTH Gym environment you can have a look at the the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) here.
# Run the ESWeek Lecture

## Follow the Gap

1. Change to the Folder 01_Follow_The_Gap

2. To experience the Follow the Gap algorithm, run the following command.
```bash
$ python3 FollowTheGap.py
```

3. You will see the simulation starting and a new windows with the simulation environment is popping up. This algorithm is running on a map that has obstacles included and you see the algorithm is avoiding these obstacles.

4. You can adjust the Follow The Gap parameter in the drivers.py file


## Follow the Raceline: Pure Pursuit

1. Change to the Folder 01_Follow_The_Gap

2. To experience the Follow the Gap algorithm, run the following command.
```bash
$ python3 PurePursuit.py
```

3. You will see the simulation starting and a new windows with the simulation environment is popping up. This algorithm is following a precalculated racline which is displayed in the simulation environment.

4. You can adjust the Pure Pursuit parameter in the drivers.py file

## Race: Graph Based Planner

1. Change to the Folder 03_GraphBasedPlanner

2. To experience the Graph based planner algorithm, run the following command.
```bash
$ python3 GraphPlanner_MultiVehiclepy.py.
```

3. You will see the simulation starting and a new windows with the simulation environment is popping up. In addition first of all the graph for the whole racetrack is created. This algorithm is following a precalculated racline which while avoiding and overtaking obstacles - fast and safe.

4. A detailed documentation and explanation of the GraphBasedPlanner can be found [here](https://graphbasedlocaltrajectoryplanner.readthedocs.io/).


# Known issues
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.11
```
And you might see an error similar to
```
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```
which could be ignored. The environment should still work without error.
