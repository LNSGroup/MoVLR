# Mice motion data

## A virtual rodent predicts the structure of neural activity across behaviours

### Mice modeling

We previously developed a skeletal model of a rat that matches the bone lengths and mass distribution of Long Evans rats1. The model has **74 DoF** and defines parent-child relationships between body parts through an acyclic tree that starts with the root (similar to the CoM) and branches to the extremities. The model has **38 controllable actuators** that apply torques to specific joints.

### Pose estimation

DANNCE (Deep Animal Pose and Kinematics Estimation)

- input: 6 cameras picture
- output: **23 3D keypoints**

### Data processing

- 3D pose estimation
  - We used **DANNCE v.1.3** to estimate the 3D pose of the animal over time from multicamera images. Pose estimation with DANNCE consists of two main steps: CoM detection and DANNCE keypoint estimation.
- CoM network training
  - We used Label3D to **manually label the rat CoM** from multicamera images in 600 frames spanning three animals. Frames were manually selected to span the range of locations and poses animals assume when in the arena. CoM networks were trained as described previously.
- DANNCE network training
  - We again used Label3D to **manually label the 3D positions** of 23 keypoints along a rat’s body. The dataset consisted of over 973 frames manually selected to sample a diverse range of poses from four different animals over eight different recordings. We finetuned a model previously trained to track keypoints in the Rat7M (https://doi.org/10.6084/m9.figshare.c.5295370.v3) dataset on our training set, as in earlier work.



So we need to use the DANNCE network to output keypoints from multicamera images.

Or manually label the 3D position by Label3D.

### Data availability

The data generated from real animals are publicly available on Harvard Dataverse, https://doi.org/10.7910/DVN/FB0MZT. To help us understand use, provide support, fulfil custom requests and encourage collaboration, we ask that users contact us when considering using this dataset. Because of their size, the data generated in simulation will be made available on reasonable request.



### Code availability

Code for all analyses will be made available from the corresponding authors on reasonable request. Repositories for skeletal registration (STAC), behavioural classification (motion-mapper) and inverse dynamic model inference are available at https://github.com/diegoaldarondo/virtual_rodent.
