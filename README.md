# ROS2-FrontierBaseExplorationForAutonomousRobot
Our autonomous ground vehicle uses Frontier Based exploration to navigate and map unknown environments. Equipped with sensors, it can avoid obstacles and make real-time decisions. It has potential applications in search and rescue, agriculture, and logistics, and represents an important step forward in autonomous ground vehicle development.

This project utilizes the **Frontier-Based Exploration** algorithm for autonomous exploration. The project employs **DFS** for grouping boundary points, **A*** for finding the shortest path, **B-Spline** for smoothing path curvature, and **Pure Pursuit** for path following, along with other obstacle avoidance techniques. The combination of these techniques aims to provide a sophisticated, efficient, and reliable solution for autonomous ground vehicle exploration in a wide range of applications.


![Screenshot_1](https://user-images.githubusercontent.com/87595266/218670694-e53bb1c4-fff2-42e9-9b9e-62b298da7fff.png)


# Youtube Project Presentation Video & Demo

https://youtu.be/UxCZAU9ZZoc

# Update Version V1.1 - 26.02.2023

https://youtu.be/_1vtmFuhl9Y

- The exploration algorithm has been optimized.

- Robot decision algorithm has been changed. Watch the video for detailed information.

- Thread structure has been added to the exploration algorithm. 


# How does it work?

1 - To get started with autonomous exploration, first launch the Map Node 

by running the following command:

`ros2 launch slam_toolbox online_async_launch.py`

2 - Then, launch the Gazebo simulation environment by setting the TurtleBot3 

model, for example, using the following command:

`export TURTLEBOT3_MODEL=burger`


`ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py`

3 - Once the simulation environment is running, run the autonomous_exploration 

package using the following command:

`ros2 run autonomous_exploration control`

This will start the robot's autonomous exploration.

## Requirements

- ROS2 - Humble
- Slam Toolbox
- Turtlebot3 Package
