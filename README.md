This is a clone of the tf-pose-estimation by Ildoo Kim modified to work with Tensorflow 2.0+!
Link to original repo: https://www.github.com/ildoonet/tf-openpose 
I integrated face detection and mediapipe Hand skelton detection with it!

![](https://img.shields.io/badge/Openpose-Pose%20Detection-red)
![](https://img.shields.io/badge/Tensorflow-Model-blue)
![](https://img.shields.io/badge/Mdiapipe-Hand%20Detection-yellowgreen)
![](https://img.shields.io/badge/Face%20Detection-Human%20Computer%20Interaction-green)
![](https://img.shields.io/badge/OpenCV-Computer%20Vision-brightgreen)

OpenPose has represented the first real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints (in total 135 keypoints) on single images.
'Openpose' for human pose estimation have been implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for real-time processing on the CPU or low-power embedded devices.
[Openpose Caffe Model Repo]( https://github.com/CMU-Perceptual-Computing-Lab/openpose)







KeyWords:

#Tensorflow  <br />
#OpenCV  <br />
#Mediapipe<br />
#Raspberry Pi<br />
#Openpose Cffe Model<br />


Results

![](https://github.com/Firos333/face_skeleton_hand_tracking/blob/master/image/Output-Skeleton1.jpg?raw=true)

![](https://github.com/Firos333/face_skeleton_hand_tracking/blob/master/image/me.png?raw=true)

![](https://github.com/Firos333/face_skeleton_hand_tracking/blob/master/image/all.png?raw=true)


Run 
```bash
git clone https://github.com/MiltekTechnologies/face-hand-pose
cd face-hand-pose
pip install --upgrade pip
pip install tensorflow
pip3 install -r requirementsupdated.txt
brew install swig
cd tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
pip install git+https://github.com/adrianc-a/tf-slim.git@remove_contrib
cd ../..
python3 threadind_f_rem.py --video=./images/me6.mp4

```


