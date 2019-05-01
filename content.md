
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


In practice, scattered data owners also demand personalized models rather than a global model for all owners. They hope to not only get help from other owners’ data to train a high accuracy model but also to gain their personalized models which can represent their unique data properties. Thus, to simultaneously address statistical and system challenges is the primary research direction of federated learning.

### General Definition of Federated Learning

Following the first federated learning paper McMahan *et al.* \[2016\], we define the objective function for the federated setting as

```latex
\begin{aligned}
F(\textbf{w}) &= \sum_{k=1}^{K} \frac{n_k}{N} F_{k}(\textbf{w}) \\
&= \sum_{k=1}^{K} \frac{n_k}{N} \frac{1}{n_k} \sum_{i \in \mathcal{P}_{k}} l(\textbf{x}_{i}, \textbf{y}_{i}; \textbf{w})
\end{aligned}
```

where $ l(\textbf{x}\_{i}, \textbf{y}\_{i}; \textbf{w}) $ is the loss function of the prediction on example $(\textbf{x}\_{i}, \textbf{y}\_{i})$ made with model parameters $\textbf{w}$, K is the total learning nodes number, $\mathcal{P}\_{k}$ is the set of indexes of data points on node k, $n_k = |P_k|$, and $\sum_{k=1}^{K} n_{k} = N$. This objective function can capture the different quantity of samples and statistical distribution of K nodes. Here, different nodes learn the global model jointly, which showing as the same loss function $l$ and parameters $\textbf{w}$.


### General Framework of Federated Multi-Task Learning

As mentioned in the introduction, federated multi-task learning is a framework that can improve the performance by directly capturing the relationships among unbalanced data in multiple devices, which implies that it can address the statistical challenges in federated learning. The general formulation for federated multi-task learning is:

```latex
\min_{\textbf{W}}\sum_{k=1}^{K}\frac{1}{n_k}\sum_{i=1}^{n_k}l_i(\textbf{x}_i,\textbf{y}_i;\textbf{w}_k) + \mathcal{R}(\textbf{W},\bm{\Omega}).
```

where $\textbf{W} = (\textbf{w}\_1,\textbf{w}\_2,...,\textbf{w}\_K) \in \mathbb{R}^{d \times{m}}$ is the parameters for different tasks and $\mathcal{R}(\textbf{W},\bm{\Omega})$ is the regularization. Different multi-task framework is mainly different from the regularization term $\mathcal{R}$.

The first term of the objective models the summation of different empirical loss of each node. The second term serves as a task-relationship regularizer with $\bm{\Omega}\in\mathbb{R}^{K\times{K}}$ being the covariance matrix Zhang and Yeung \[2012\].

The covariance matrix is able to describe positive, negative and unrelated correlation between nodes, which can either known as priori or being measured while learning the models simultaneously.

Each element $\bm{\Omega}\_{i,j}$ is a value that indicates the similarity between two nodes. Here we use a bio-convex formulation in Zhang and Yeung \[2012\], which is a general case for other regularization methods,


```latex
\mathcal{R}(\textbf{W},\bm{\Omega}) = \lambda_ 1tr(\textbf{W}\bm{\Omega}^{-1}\textbf{W}^{T}) + \lambda_2||\textbf{W}||_F^2.
```

where we constrain $\vec{W}$ with covariance matrix $\bm{\Omega}^{-1}$ through matrix trace $tr(\bm{W}\bm{\Omega}^{-1}\textbf{W}^{T})$. This means the closer $\textbf{w}_i$ and $\textbf{w}_j$ is, the larger the $\bm{\Omega}\_{i,j}$ will be. Specifically if $\bm{\Omega}$ is an identity matrix, then each node is independent to each other.
Smith *et al.* \[2017\] proposed MOCHA based on the above multi-task learning framework. However, MOCHA can only handle convex functions in federated multi-task learning settings, which can not be generated to non-convex deep learning models. Our work generates federated multi-task learning framework to the non-convex DNN setting.


### Federated Multi-Task Deep Learning Framework

DNNs are able to extract deep features from raw data. However, to the best of our knowledge, DNNs has not been applied to federated multi-task problems. We thus consider DNNs as our feature transformation function and make prediction based on the hidden features. Formally speaking, the formulation can be defined as:

```latex
\begin{aligned}
& \begin{aligned} \min_{\bm{\theta}, \vec{U},\vec{W},\bm{\Omega}}
&\sum_{k=1}^{K}\frac{1}{n_k}\bigg[\bigg.\sum_{i=1}^{n_k}l(f(\vec{x}_i^{k},\bm{\theta}_k,\vec{U}_k,\vec{w}_k),\vec{y}_i^k) \\
&+ \frac{1}{2}\lambda_ 1tr(\vec{W}_k\bm{\Omega}_k^{-1}\vec{W}_k^{T})\bigg]\bigg. + \frac{1}{2}\lambda_2||\vec{W}||_F^2\\
&+\frac{1}{2}\lambda_3||\bm{\theta}||_F^2 + \frac{1}{2}\lambda_4||\vec{U}||_F^2, \end{aligned} \\
& \quad \text{s.t.} \quad  \bm{\Omega}_k \ge 0 \quad \text{and} \quad tr(\bm{\Omega}_k) = 1, \quad k = 1, 2, ... ,K.
\end{aligned}
```

where $f(\cdot)$ represents DNNs feature mapping as shown in Figure \_\_. $\bm{\theta}\_k$ is the feature transformation network. $\bm{U}\_k$ and $\vec{w}\_k$ are output layer (e.g. softmax). The first constraint in \_\_ holds due to the fact that $\bm\Omega$ is defined as a task covariance matrix. The second constraint is used to restrict its complexity.

In federated  learning situation, training should be conducted on each node respectively. One intuitive thought is the centralized network topology in McMahan *et al.* \[2016\], where one center node synchronously takes a weighted average parameters of every clients at each time step (Figure \_\_). However,  this model faces the problems that in DNNs situation, far more parameters need to be calculated and transferred. Each node has heterogeneous computing performance and network bandwidth (Figure \_\_). Setting one center node to synchronously collect all the parameters will induce high communication cost and low convergence speed. In order to overcome these problems, we design a decentralized topology, where each node only needs to share their parameters with neighbored nodes as shown in Figure \_\_, where there is no communication between worker one and worker 4. Abandoning the central node induces the problem that parameters cannot be exchanged and synchronized amongst every nodes, which means that the centralized optimization method can not be achieved on this topology. To this end, we propose a Decentralized Periodic Averaging SGD (DPA-SGD) to tackle the optimization problem in decentralized topology.

## Approach

### Decentralized Periodic Averaging SGD

As for decentralized topology, due to the disappearing of central node, same central averaging method can not be applied. In order to overcome this problem, we come up with a novel optimization method, Decentralized Periodic Averaging SGD (DPA-SGD). The main idea of DPA-SGD is that during the communication period $\tau$, local SGD is applied on each node respectively, and synchronizing all the parameters at every $\tau$ iterations amongst its connected neighbors. Due to this decentralized diverse connection, one global $\bm\Omega$ can not represent the individual correlation. So we propose to use a distinct covariance matrix $\bm{\Omega}\_k$ to represent their own mutual relationship. We also come up with an effective way to update the different $\bm\Omega_k$. To be specific, consider one particular node $m$ and its neighbor connected nodes as set $\mathcal{M}$.

The new objective function can be defined as:

```latex
\begin{aligned}
& \begin{aligned} \min_{\bm{\theta}, \vec{U},\vec{W},\bm{\Omega}}
&\sum_{k=1}^{K}\frac{1}{n_k}\bigg[\bigg.\sum_{i=1}^{n_k}l(f(\vec{x}_i^{k},\bm{\theta}_k,\vec{U}_k,\vec{w}_k),\vec{y}_i^k) \\
&+ \frac{1}{2}\lambda_ 1tr(\vec{W}_k\bm{\Omega}_k^{-1}\vec{W}_k^{T})\bigg]\bigg. + \frac{1}{2}\lambda_2||\vec{W}||_F^2\\
&+\frac{1}{2}\lambda_3||\bm{\theta}||_F^2 + \frac{1}{2}\lambda_4||\vec{U}||_F^2, \end{aligned} \\
& \quad \text{s.t.} \quad  \bm{\Omega}_k \ge 0 \quad \text{and} \quad tr(\bm{\Omega}_k) = 1, \quad k = 1, 2, ... ,K.
\end{aligned}
```

where $\vec{W}\_k = (\vec{w}\_1,\vec{w}\_2,...,\vec{w}\_m,...\vec{w}\_{|\mathcal{M}|})\in\mathbb{R}^{d\times|\mathcal{M}|}$ is the parameters for $m$ and its neighbor tasks. The matrix $\bm{\Omega}\_k\in\mathbb{R}^{|\mathcal{M}|\times{|\mathcal{M}|}}$ represents the correlation amongst nodes in set $\mathcal{M}$. Here in order to record the entire nodes connection in the network, we introduce a *node connection matrix* $\vec{M}\in\mathbb{R}^{K\times{K}}$ represents the neighbor relationships for each nodes, where $M_{i, j}$ is a value that indicates node $i$ and $j$ are connected as shown in Figure \_\_, where worker one is only connected with worker two and four. Note that, if $\vec{M} = \vec{I}$ (Identity matrix), then every nodes are independent and update the parameters respectively. If $\vec{M} = \vec{J}$ (one for each element), the model is degenerated into centralized model. We study the model performance under sparse matrix $\vec{M}$ and find that similar results can be achieved as a ring network topology, which each node is only connected with its nearby two nodes, as illustrated in Figure \_\_.

To solve this non-convex problem, we apply the alternating optimization method  Zhang and Yeung \[2012\], where alternately updating parameters $\vec{X} = (\vec{W, U},\bm{\theta})$ and $\bm\Omega$.

*Optimizing* $\bm\theta, \vec{W}$ and $\vec{U}$: For simplicity, we define set $\Xi = (\bm\Omega_1, \bm\Omega_2,...,\bm\Omega_K)$ to represent the correlation matrix for every nodes. Fixing $\Xi$, we can use SGD method to update $\bm\theta, \vec{W}$ and $\vec{U}$ jointly. Our problem can then be reformulated as:

```latex
 G(\vec{W}, \vec{U}, \bm\theta|\Xi) = \sum_{k=1}^{K}\frac{1}{n_k} \bigg[\bigg. \sum_{i=1}^{n_k}l(f(\textbf{x}_i^{k},\bm{\theta}_k,\textbf{U}_k,\textbf{w}_k),\textbf{y}_i^k)\\ + \frac{1}{2}\lambda_1tr(\textbf{W}_k\bm{\Omega}_k^{-1}\textbf{W}_k^{T})\bigg]\bigg.+ \frac{1}{2}\lambda_2||\textbf{W}||_F^2\\
+ \frac{1}{2}\lambda_3||\bm{\theta}||_F^2 + \frac{1}{2}\lambda_4||\textbf{U}||_F^2
```

We can calculate the gradient of $\vec{W}$, $\vec{U}$ and $\bm\theta$ respectively. Let $L = \sum_{k=1}^{K}\frac{1}{n_k}\sum_{i=1}^{n_k}l(f(\bm{x}\_i^{k}, \bm{\theta}\_k,\bm{u}\_k,\bm{w}\_k),\bm{y}\_i^k)$. Then the gradient formulations for each node are:

```latex
\frac{\partial{G(\vec{W}, \vec{U}, \bm\theta|\Xi)}}{\partial{\vec{w}_k}} = \frac{\partial{L}}{\partial{\vec{w}_k}} + \lambda_1\sum_{i=1}^{\mathcal{|M|}}\frac{1}{n_i}{\vec{w}_k}\bm\Omega_i^{-1}+ \lambda_2\vec{w}_k
```
where the summation is amongst all the nodes connected to node $k$,

```latex
\frac{\partial{G(\vec{W}, \vec{U}, \bm\theta|\Xi)}}{\partial \bm\theta_k} = \frac{\partial{L}}{\partial\bm\theta_k} + \lambda_3\bm\theta_k,
```

```latex
\frac{\partial{G(\vec{W}, \vec{U}, \bm\theta|\Xi)}}{\partial{\vec{u}_k}} = \frac{\partial{L}}{\partial{\vec{u}_k}} + \lambda_4\vec{u}_k
```

*Optimizing* $\Xi$: In paper Zhang and Yeung \[2012\], an analytical solution form is given for $\bm\Omega$:

```latex
\bm\Omega = \frac{(\vec{W}^T\vec{W})^\frac{1}{2}}{tr((\vec{W}^T\vec{W})^{\frac{1}{2}})}
```

Apparently, if $\bm{w}\_i$ and $\bm{w}\_j$ are close to each other, $\bm{\Omega}$ will be large. However, the missing central node forbidding to average parameters globally. So here we propose a novel way to update each $\bm\Omega_k\in\Xi$:

```latex
\bm\Omega_{t+1}^{(k)} \leftarrow \eta\frac{1}{|\mathcal{M}|}({\sum_{i=1}^{|\mathcal{M}|}}\frac{1}{n_i}\bm\Omega_{t}^{(i)} + \frac{(\vec{W}_k^T\vec{W}_k)^\frac{1}{2}}{tr((\vec{W}_k^T\vec{W}_k)^{\frac{1}{2}})})
```

The first averaging term can incorporate the nearby nodes correlation into its own and the second term captures the new correlation between its neighbors as shown in Figure \_\_.


### Algorithm

In general, the algorithm of DPA-SGD can be summarized as: while in local update period, each node calculates the gradient $g(\vec{X}\_t^{(i)})$ based on one mini-batch of data and then update $\vec{X}^{(i)}$; For every synchronization per $\tau$ update, the novel update way of $\bm\Omega$ is conducted.

**\<Replace with Algorithm\>**

## Comparison to the past works

Here we illustrate system-wise advantages of DPA-SGD:

**\<Replace with Advantage Fig\>**

**Faster convergence speed**. Figure \_\_ illustrates three reasons that DHA-SGD can speed up convergence.
1. *Periodic averaging* can alleviate the communication delay by reducing times of synchronization which only happen periodically. As we can see in figure 4, the yellow blocks (communication) will largely be deleted due to the periodic averaging mechanism.
1. This idle time can also be significantly reduced through periodic averaging as shown in Figure 4.
1. In the decentralized topology, because a worker only needs to exchange gradients with its neighbors, another worker with slow computation and communication will not interrupt its iteration. For example, worker 2 in the above figure can synchronize earlier without waiting for worker 4. Thus, DHA-SGD can largely reduce convergence time.

**Fewer communication rounds**. The periodic averaging strategy can reduce the number of communication rounds. Although our work focuses on optimizing the ratio of computation and communication rather than improving the communication cost for each iteration, gradient compression method (Deep Gradient Compression \[Lin *et al.*, 2017\]) for each iteration can directly extendable to to our algorithm in a practical system.

## Results

- Especially if you approach and final results aren’t
  impressive, please show us your progressive steps

```latex
\begin{aligned}
y &= y(x,t) = A e^{i\theta} \\
&= A (\cos \theta + i \sin \theta) \\
&= A (\cos(kx - \omega t) + i \sin(kx - \omega t)) \\
&= A\cos(kx - \omega t) + i A\sin(kx - \omega t)  \\
&= A\cos \Big(\frac{2\pi}{\lambda}x - \frac{2\pi v}{\lambda} t \Big) + i A\sin \Big(\frac{2\pi}{\lambda}x - \frac{2\pi v}{\lambda} t \Big)  \\
&= A\cos \frac{2\pi}{\lambda} (x - v t) + i A\sin \frac{2\pi}{\lambda} (x - v t)
\end{aligned}
```


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
