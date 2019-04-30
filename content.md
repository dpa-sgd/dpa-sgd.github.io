Federated Learning enables training models collaboratively over a large number of distributed edge devices without integrating their local data, while Federated Multi-Task Learning can further help to learn a personalized model for each device.
However, they both pose particular statistical and systems challenges. To simultaneously address these two challenges, and focusing on training deep neural networks models collaboratively, we propose a decentralized approach with the framework and optimization co-design.

## Problem formulation
### Statistical Challenges
1. **Non-IID**: Each worker generates data in a non-i.i.d. (independent and identically distributed) manner with a distinct statistical distribution.
1. **Unbalanced Local Data**: Workers have different quantity of data sample due to their different behaviors.

These two characteristics bring challenges to learning a high-performance personalized model for each worker.

### System Challenges
1. **Larger Worker Number**: The worker number is typically larger than cloudbased learning. The larger number will pose a higher communication cost and difficult communication
synchronization.
1. **Heterogeneous Networks**: Each worker may differ in communication capacity due to heterogeneous networks(4G, WiFi, and other IoT protocol).
1. **Heterogeneous Computation**: Computational
capacities of each worker may differ due to variability in hardware(CPU, memory).

These three characteristics make communication cost and low-speed training become a major bottleneck towards real-world deployment.


In practice, scattered data owners also demand personalized models rather than a global model for all owners. They hope to not only get help from other owners’ data to train a high accuracy model but also to gain their personalized models which can represent their unique data properties. Thus, to
simultaneously address statistical and system challenges is the primary research direction of federated learning.

## Approach
```latex
\sum_{i=1}^n i^3 = \left( \frac{n(g(n)+1)} 2 \right) ^2 \bm\Omega
```


## Comparison to the past works

- What has been done before

- What is new about your problem or approach

  - In other words, why is your work cool?

## Results

- Especially if you approach and final results aren’t
  impressive, please show us your progressive steps


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
