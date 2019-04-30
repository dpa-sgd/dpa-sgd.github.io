Furniture assembly is a difficult challenge even for humans, requiring excellent planning and manipulation skills. Recently, reinforcement learning has acheived notable advances with the advent of deep learning, and its application to executing realistic tasks for robotics is promising. However, current robotic simulations and benchmarks focus on low-level control tasks with a short time horizon like grasping and picking. To accelerate the application of reinforcement learning to real world tasks, we propose the **IKEA Furniture Assembly Environment**,  a realistic environment that models the task of furniture assembly to learn and benchmark complex manipulation skills with hierarchical structures and a long time horizon.

Our environment features over 80 furniture models including chairs, tables, cabinets, bookcases, desks, shelfs, and tv units. To help transfer learning from simulated to real-world environments, all the furniture models are created following the IKEA’s official user’s manuals with a minor simplification in small details such as carving and screws. Simple furniture only have 2 parts while complex furniture have up to 13. The furniture models are composed of parts and joints that connect them. We allow the user to relax or constrain conditions to control the difficulty, such as distance and angle between joints or requiring a screw. We offer a tool in Unity to automatically create and edit joint positions and constraints for ease of use.
![Example Furniture models](/img/furniture_models.png "Furnitures")

Our environment supports a diverse set configurations for realistic simulation. We provide variability in furniture compositions, visual appearance of parts, part shapes and textures, physics, lighting condition, background scenes and more.
||||
|:---:|:---:|:---:|
|![Indoor Enviroment 1](/img/env_var/indoor_wood_floor.png "indoor_wood_floor")|![Indoor Enviroment 2](/img/env_var/indoor_wood_floor2.gif "indoor_wood_floor2")|![Lab Enviroment](/img/env_var/indoor_lab.gif "indoor_lab")|

### Diverse Robots, Actions, and Observations
To assemble furniture, the agent must select a part, align the part with the corresponding part, and attach the two parts together. We offer various types of robots, observations and actions to vary the difficulty of the problem. For example, the agent can be a fully functional Baxter robot.  By default, the Baxter is controlled through joint position control where the policy outputs a target angle and force for each joint. The Baxter can also be controlled with inverse kinematics to abstract the movement. Currently, the environment supports the Baxter and Sawyer robots.
| ||
|:---:|:--:\
|![Baxter](/img/Baxter_Move_Rotation.gif =50%x* "baxter")|![Sawyer](/img/Baxter_Move_Rotation.gif =50%x* "baxter")|

Because learning continuous control in robotics can be challenging, the cursor agent abstracts away the robot control to one or two 3D cubes that serve as cursors. In the continuous control case, the agent generates 6 floating point values for each cursor to move and rotate the cursor by the value amount in the X, Y, Z axes. The agent also needs to output a value above 0 to select anything within the cursor, and otherwise it will drop whatever is in the cube. Finally, to attach 2 parts together, the agent should output a value above 0 in the match dimension to signal a match.

|||
|:---:|:---:|
|![One cursor](/img/chair15_2p_2x.gif "Table TID=28 With 1 Pointer")|![Two cursors](/img/chair15_2p_2x.gif =50%x* "Chair TID=15 With 2 Pointers")|

<!-- ![Cursor's action space](/img/cursor_action_space.png =50%x50% "Cursor's action space") -->

In the discrete case, the movement and rotation are discretized into 6 cardinal directions, e.g move up / down / left / right / forward / backward. The policy just needs to output one command at each time step. An example trajectory could be move up, select, rotate left, match.



|||||
|:---:|:---:|:---:|:---:|
|![Current](/img/current.png)|![Parts](/img/parts.png)|![Goal](/img/goal.png)|![Segmentation](/img/segmentation.png)|

The observation space can be composed of the following:
- __True state__:  coordinate positions and rotations of all objects relative to the agent
- __Current observation__: RGB image of the environment 
- __Parts observations__: RGB image of each furniture part
- __Goal observation__: RGB image of the goal configuration
- __Segmentation mask__: mask over the current observation to show where objects are 
- __Agent state__: agent's internal state
    - Baxter agent's internal state contains the positions of all joints and rotations relative to the base of the robot.
    - Cursor agent's internal state contains the XYZ coordinates of each cursor.



<!--After training, the environment will test the robot on unseen furniture configurations. The environment provides variability in compositions, visual appearances, object shapes, and physical properties. The environment contains a diverse set of furniture including chair, table, cabinet, bookcase, desk, shelf, and tv unit; and it supports changes in light condition, background, texture, and colors. For a robot to generalize this tasks well, it has to master a hierarchy of skills that are curricula to autonomous manipulation:
- Visual sensing
  - Test robots’s ability to understand 3D scenery
  - The robots must identify the shape of each 3D furniture part and understand the 3D structure of the goal furniture to know where a furniture part belongs in the goal furniture
- Step by step planing
  - Test robots’s ability to make complex sequential decisions with a long horizon
  - The robots must deduce the order of  how to assembling each parts from either the manual or the goal furniture
-  Low level control
  - Test robot’s ability to conducting low level manipulation tasks like picking, placing and inserting
  - Robot must be able to manipulate the furniture pieces precisely to assemble them by connecting joints

# Some Details
- Implementation
  - The Unity3D game engine with ML Agent is used as a back-end framework to support realistic rendering and fast simulation
  - To help transfer learning from simulated to real-world environments, all the furniture models are created following the IKEA’s official user’s manuals with a minor simplification in small details such as carving and screws
- In addition to the observation, the environment can provide the final configuration, as well as the furniture assembly instruction in the form of intermediate observations of furniture construction
- Robot Agent
  - The environment contains two robotic arms and the arms can assemble furniture by repeating the following process: picking two pieces, moving them toward each other, aligning two pieces, and attaching them
  - The environment will support abstraction for robot control including high-level action (e.g., pick A, attach A and B), discrete control (e.g., move forward, rotate clockwise), and joint velocity control
  -->

----

# Feature list
text

----

# Baselines
text

----

# Citations
```
@inproceedings{
  lee2019ikea,
  title = {IKEA Furniture Assembly Environment},
  author = {Lee, Youngwoon and Hu, Edward S and Zhengyu, Yang and Lim, Joseph J},
  year = {2019},
  url={https://youngwoon.github.com/assembly},
}
```
