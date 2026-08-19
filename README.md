# Project SY27 ZOE - Team Nowaymo


## Launch

Go to `worskpace/`, then build and source using:
```bash
colcon build --symlink-install
source install/setup.bash
```
Then launch the project:
```bash
ros2 launch manager manager.launch.xml offline:=false
```

This will launch the Control, Navigation and Interior Perception nodes. A calibration is made to recognize your face and later tell if you are looking at the road or not (in which case the car will stop). You will be able to drive the car by doing gestures with your hand. You need to hold the gesture for at least 1 or 2 seconds but don't need to keep more than that if the program has validated your choice. Here are the gestures:
- Thumb up: Start the car
- Full palm: Stop the car
- Thumb to the left: Go left (stay in roundabout)
- Thumb to the right: Go right (go out of roundabout)

In another terminal, open RViz (`rviz2`), and open the config `misc/rviz_config.rviz`. You should see the car moving along the path if you are driving, and information about the driver attention span and hands.

To use Exterior Perception and Localisation nodes, uncomment their launch in `workspace/src/manager/launch/manager.launch.xml` or launch them separately using
```bash
ros2 launch PKG PKG.launch.xml # PKG = perc_ext or localisation
```

You can also write `params.yaml` to changes the controlers parameters:
```bash
ros2 param load /control_node params.yaml
```


## ROS tutorial
### How to launch your package
#### Create a launch.xml file

Let PACKAGE be your package name. In `src/PACKAGE/`, create a folder named `launch`. Now, in `src/PACKAGE/launch/`, create a file named `PAGACKE.launch.xml` (replacing PACKAGE with your package name). In this file, add the following, replacing NODE_NAME and PACKAGE accordingly:

```xml
<launch>
    <node name="NODE_NAME" pkg="PACKAGE" exec="NODE_NAME" output="screen" />
</launch>
```

#### Modify setup.py
Now, in your `setup.py` file (located at `src/PACKAGE/setup.py`), add the following inside data_files (do NOT replace "package_name" with your actual package name):
```
(f"share/{package_name}/launch", glob("launch/*.launch.xml")),
```
Be sure to import `glob` at the top of your file if it's not already there: `from glob import glob`

After that, add an entry_point, replacing python_file with your actual python_file and the rest accordingly:
```py
"console_scripts": [
        "NODE_NAME = PACKAGE.python_file:main",
    ],
```

#### Add your launch to manager.launch.xml
In `src/manager/launch/manager.launch.xml`, add the following line after the comment "Put your nodes here" around line 41, replacing PACKAGE with your package name (it's locating and including your launch file you just made):
```xml
<include file="$(find-pkg-share PACKAGE)/launch/PACKAGE.launch.xml" />
```

#### Add shared documents
In `src/shared` is installed all shared datas (enums, files, ...). To be sure to have those values in your module, you have to add in your `package.xml` : 
```xml
<depend>shared</depend>
```

Now, you can in your project import file from it (example to import the file "enum_segment_type"):
```py
from shared.enums.enum_segment_type import ...
[...]
```

#### Build and source
Now build and source in the terminal and when launching the manager package with `ros2 launch manager manager.launch.xml`, your package should launch alongside the manager package!


### How to use bags

Download the bags you want to use and put it in `workspace/src/bagfiles`. If the folder doesn't exist, create it, it's already in the gitignore to avoid pushing bags on git. If your bag contains a metadata file (`metadata.yaml`), keep it. For each bag, put all files in a separate folder in `bagfiles/`. This folder will be your bag.

Launch your bag with:

```bash
cd workspace/src/bagfiles                       # Go to bagfiles/
ros2 bag play your_bag_folder/                  # Play your bag
ros2 bag play --start-paused your_bag_folder/   # Play your bag but better
```

Exemple: consider the following bags:
```bash
bagfiles/
├── sy27_road_ground_truth2
│   ├── metadata.yaml
│   └── sy27_road_ground_truth2_0.mcap
└── sy27_road_ground_truth_live_0
    ├── metadata.yaml
    └── sy27_road_ground_truth_live_0.mcap
```

You would launch your bags with:
```bash
ros2 bag play sy27_road_ground_truth2/
ros2 bag play sy27_road_ground_truth_live_0/
```

Once your bag is started, you can then show the topics, and use RViz, RQT, or topic echo to see topics content using the commands listed below.

### Useful commands

```bash
ros2 topic list                      # List all topics
ros2 topic type topic_name           # Show message type on this topic
ros2 topic echo topic_name           # Print the content of this topic in real time
ros2 topic echo --no-arr topic_name  # Same but cleaner printing
ros2 topic info --verbose topic_name # Print type, publishers and subscribers of the topic

rviz2   # Launch RViz (3D)
rqt     # Launch RQT (2D)

ros2 launch PACKAGE LAUNCHNAME.launch.xml   # Launch a package
ros2 launch manager manager.launch.xml            # Exemple with manager package
```