# Analysis of Running Form with Keypoint R-CNN
Webapp that uses a Human Point Estimation model from Pytorch to analyse your running form

# Description
This webapp uses Keypoint R-CNN from Pytorch to analyse side profile images of runners. Over 100 side-profile photos of Eliud Kipchoge (2-time Olympic Marathon gold medallist, and the only human in history to have completed a marathon in less than 2 hours) were analysed to determine his knee angle and trunk angle. These are 2 important aspects of running form.

## Knee Angle
Knee angle is used to detect the presence of overstriding, which occurs when the front leg extends too far out in front of the body when it strikes the ground. Overstriding generates a braking force and increases impact on the knee, which may increase the risk of injury.

## Angle of Forward Lean
The trunk angle is also important, as ideal running mechanics call for a specific range of forward trunk lean. A small change in posture can provide large impacts in terms of muscle activation, shock absorption, and overall injury risk. Use this website to compare your running form with Eliud Kipchoge's, and check if you're running with the right knee angle and forward trunk lean!

# Installation 
From the main directory, open a terminal and run ```pip install -r requirements.txt```.

# How To Use
From the main directory which contains ```app.py```, open a terminal and run ```flask run```. Click on the link in the terminal's output to load the webpage. To use it, simply upload a side-profile image of yourself running. You can record a video of yourself running, and take a screenshot at the exact moment when either foot strikes the ground. You can be facing either left or right in the photo. Refer to the sample photos below for some examples. For better results, ensure that the uploaded photo is of decent resolution, and ensure that there is only one person in the photo. After uploading, please wait for a few moments, you will be redirected to another page.
