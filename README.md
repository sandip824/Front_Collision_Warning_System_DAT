# Front_Collision_Warning_System using TFOD 2.0
Aim of this work is to impliment a  front collision warning system , using __tensorflow objetc detection API__. 
Front Collision Warning(FCW) system is an ADAS(Advance Driver Assistance Syatem). With the current advancements in the area of computer vision. we have efficent object detection APIs available as open source. This work is for just to demostrate the ability of object detection to be used in FCW feature in ADAS systems.


## Installation
The first and most important task is to install tensorflow object detection API in you system.

### Step 1 :TFOD installation
follow the steps mentioned [here](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b) in order to install object detection API in your system.
PLease make sure that TFOD is working on your system by checking run tutorial mentioned in the link abobe 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages mentioned in requirment.txt.

### Step 2 :Install dependencies
```bash
pip install -r requirment.txt
```

## Implimentetion Steps
This project is implimeted in two steps 
### 1. Object detection
### 2. FCW conditioning 
The FCW warning condtion is desided on below factors .
* Find distance of the object from vehicle 
* Find apex ratio to deside collison warnig threshold 
* Collision Time 
* Vehichle Speed 

This project is implimented in two distinct phases




## Results.


https://user-images.githubusercontent.com/47384889/153769926-9d4213f8-5152-40f6-a5ea-0fc067d44147.mp4


## Challanges faced.
list down the challanges here

## Credits
This project uses Open Source components. You can find the source code of their open source project along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

* Installing TensorFlow Object Detection API on Windows 10 [here](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b)
* Determining other vehicle distances and collision warning - Self Driving Cars in GTA [here](https://pythonprogramming.net/detecting-distances-self-driving-car/)
