# ROS2-FrontierBaseExplorationForAutonomousRobot
Our autonomous ground vehicle uses Frontier Based exploration to navigate and map unknown environments. Equipped with sensors, it can avoid obstacles and make real-time decisions. It has potential applications in search and rescue, agriculture, and logistics, and represents an important step forward in autonomous ground vehicle development.


![Screenshot_1](https://user-images.githubusercontent.com/87595266/218670694-e53bb1c4-fff2-42e9-9b9e-62b298da7fff.png)


# Youtube Project Presentation Video & Demo

https://youtu.be/UxCZAU9ZZoc


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

## Requirements

- ROS2 - Humble
- Slam Toolbox
- Turtlebot3 Package