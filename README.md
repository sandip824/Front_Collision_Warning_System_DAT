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
* Apex ratio to deside collison warnig threshold 
* Distance of the object from vehicle 
distance calculation is done in 2 steps 
    1. Camera calibration to find focal length of camera.
       To find the foclal lemngth of the camera a object with known weidth is kept at known istance. Then using triangal similarity model of pinhole camera
       the focal length is calculated as 
       f = (d x r) / R
    2. Actual distance calculation using focal length.
      Once we calculate the focal lengt if the camera its time to calculate real distance of the object from camera.
      Distance of the object is calculated as 
      d = (f X R)/r
      
      where 
      f - focal length of camera
      R- Actual weidth of the object 
      r- weidth of the detected object in image.
      Note : Please note that here we have considered real weidth od the object as average weidth of passanger cars 
      so the whatever distance will be calculated will be approximate not the accurate distance.
      
![images](https://www.researchgate.net/profile/Oezge-Bayri-Oetenkaya/publication/333457836/figure/fig1/AS:764001881432064@1559163651059/Figure-8-Focal-length-rule-But-in-this-case-we-cannot-examine-the-exact-value-of-focal.png)
[image Source](https://www.researchgate.net/profile/Oezge-Bayri-Oetenkaya/publication/333457836/figure/fig1/AS:764001881432064@1559163651059/Figure-8-Focal-length-rule-But-in-this-case-we-cannot-examine-the-exact-value-of-focal.png)

* __Vehichle Speed:____
      Ideally vehicle speed will be calculated using sensors installed on the vehicle but in our case its not avaible, So i have tried to simulated the vehicle speed           calculation by calculating SSIM between two subsequent video frames. Please note that this speed calculation is not accurate it is just for simulation purpose.
      Calculating vehicle speed from the video sequence is still a research topic. 
* __Collision Time:__
      The collision time for every detected object is calculated by deviding distance with speed
      Collision time = (Object distance/Speed of vehicle)

## Results.
https://user-images.githubusercontent.com/47384889/154838676-5b46f305-7d77-4db6-adb4-45af2cad46cc.mp4

## Limitations

## Credits
This project uses Open Source components. You can find the source code of their open source project along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

* Installing TensorFlow Object Detection API on Windows 10 [here](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b)
* Determining other vehicle distances and collision warning - Self Driving Cars in GTA [here](https://pythonprogramming.net/detecting-distances-self-driving-car/)
* Collision_Warning_Based_on_Multi-Object_Detection_and_Distance_Estimation [here](https://www.researchgate.net/publication/348155370_Collision_Warning_Based_on_Multi-Object_Detection_and_Distance_Estimation)
