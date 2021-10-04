## ESweek2021_educationclassA3
This is the repository that includes the code material for the ESweek 2021 for the Education Class Lecture A3 "Learn to Drive (and Race!) Autonomous Vehicles"

## Environment Installation guide

1. Create an Virtual Environment with Python 3.8
2. Install the required python packages with the following command:

```bash
pip3 install -r requirements.txt
```
3. Install the F1TENTH gym environment while in the root folder of this repository by running the following command:
```bash
$ pip3 install -e gym/
```

## Run the ESWeek Lecture

# Follow the Gap

# Follow the raceline: Pure Pursuit


## Known issues
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
