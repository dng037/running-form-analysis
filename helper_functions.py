import math
import torch
import torchvision
from torchvision import transforms as T
import cv2
import numpy as np
import statistics

# Pretrained Pytorch model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# Check if gpu is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Send model to GPU
model.to(device)
# Put the model in inference mode
model.eval()

# Create the list of keypoints
points = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

OVERSTRIDING = """From the 100+ side-profile photos of Eliud Kipchoge that were analyzed, his smallest stride angle was 83.67 degrees. Your stride angle is less than that, which indicates that you're likely overstriding. To fix this, try
 increasing your cadence, and extending your stride behind your body rather than in front of it. You can also stretch your hip flexors to increase the range of motion of your hips."""

GOOD_STRIDE = """From the 100+ side-profile photos of Eliud Kipchoge that were analyzed, his smallest stride angle was 83.67 degrees. Your stride angle is greater than that, which means that your stride angle is good! 
Maintain the good form, and remind yourself to land with your feet directly beneath you instead of out in front."""

GOOD_LEAN = """From the 100+ side-profile photos of Eliud Kipchoge that were analyzed, his trunk angle ranged from 62.11 to 73.09 degrees. Your trunk angle lies within that range, well done!"""

TOO_UPRIGHT = """From the 100+ side-profile photos of Eliud Kipchoge that were analyzed, his trunk angle ranged from 62.11 to 73.09 degrees. Your trunk angle is greater than that, which indicates that you're running too upright. 
Try to lean forward slightly, without rounding your lower back or hunching your shoulders. If you're also overstriding, leaning forward slightly more might help to fix that!"""

TOO_FORWARD = """From the 100+ side-profile photos of Eliud Kipchoge that were analyzed, his trunk angle ranged from 62.11 to 73.09 degrees. Your trunk angle is less than that, which indicates that you're leaning too far forward. 
Try to run tall: imagine that you're leaning into a strong wind, or that you're a puppet with a string connected to the top of your head, which is pulling your head up and your chest out as you run!"""

EXPLANATIONS = [OVERSTRIDING, GOOD_STRIDE, GOOD_LEAN, TOO_UPRIGHT, TOO_FORWARD]

def get_model_output(image_bytes):
    # Image pre-processing
    transform = T.Compose([T.ToTensor()])
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    img_tensor = transform(image)
    img_tensor = img_tensor.to(device)

    # Forward-pass the model
    output = model([img_tensor])[0]

    return output

def draw_selected_keypoints_per_person(image_bytes, all_keypoints, all_scores):
    # To return
    return_dict = {}
    
    # Create a copy of the image
    img_copy = img_copy = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    # Used to store the mean scores for the important joints, for each detected person
    mean_scores = []
    
    # Iterate through every person detected
    for person in range(len(all_keypoints)):
        # All scores
        scores = all_scores[person, ...]
            
        # Get the corresponding scores for the important points
        right_knee = scores[points.index('right_knee')].item()
        right_ankle = scores[points.index('right_ankle')].item()
        left_knee = scores[points.index('left_knee')].item()
        left_ankle = scores[points.index('left_ankle')].item()
        nose = scores[points.index('nose')].item()
        left_hip = scores[points.index('left_hip')].item()
        right_hip = scores[points.index('right_hip')].item()
        
        # Getting the mean of the scores for the chosen points
        mean = statistics.mean([right_knee, right_ankle, left_knee, left_ankle, nose, left_hip, right_hip])
        
        mean_scores.append(mean)
    
    # Find the person with the highest mean score for the important points
    mean_scores = np.array(mean_scores)
    person_id = np.argmax(mean_scores)
    
    # Get the keypoint locations for the detected person
    keypoints = all_keypoints[person_id, ...]

    # Get the important points
    right_knee = keypoints[points.index('right_knee')]
    right_ankle = keypoints[points.index('right_ankle')]
    left_knee = keypoints[points.index('left_knee')]
    left_ankle = keypoints[points.index('left_ankle')]
    nose = keypoints[points.index('nose')]
    left_hip = keypoints[points.index('left_hip')]
    right_hip = keypoints[points.index('right_hip')]
    hip = None
    
    # Check which direction kipchoge is facing
    if nose[0].item() > left_hip[0].item() or nose[0].item() > right_hip[0].item():
        return_dict['direction'] = "right"
        hip = right_hip
    else:
        return_dict['direction'] = "left"
        hip = left_hip
    
    # Update nose and hip for lean
    return_dict['nose'] = (nose[0].item(), nose[1].item())
    return_dict['hip'] = (hip[0].item(), hip[1].item())
    
    right = None
    
    # If right ankle is lower to the ground than left ankle
    if right_ankle[1].item() > left_ankle[1].item():
        # If right ankle and right knee are close enough in x coordinates
        if abs(right_ankle[0].item() - right_knee[0].item()) < 50:
            right = 1
        else:
            right = 0
    # If left ankle is lower to the ground than right ankle
    else:
        # If left ankle and left knee are close enough in x coordinates
        if abs(left_ankle[0].item() - left_knee[0].item()) < 50:
            right = 0
        else:
            right = 1
    
    # If right foot is striking the ground
    if right == 1:
        # Update ankle and knee for stride
        return_dict['ankle'] = (right_ankle[0].item(), right_ankle[1].item())
        return_dict['knee'] = (right_knee[0].item(), right_knee[1].item())

    # If left foot is striking the ground
    else:
        # Update ankle and knee for stride
        return_dict['ankle'] = (left_ankle[0].item(), left_ankle[1].item())
        return_dict['knee'] = (left_knee[0].item(), left_knee[1].item())
                                  
    # Draw the circles
    cv2.circle(img_copy, (int(return_dict['ankle'][0]), int(return_dict['ankle'][1])), 15, (255,0,0), -1)
    cv2.circle(img_copy, (int(return_dict['knee'][0]), int(return_dict['knee'][1])), 15, (255,0,0), -1)
    cv2.circle(img_copy, (int(return_dict['nose'][0]), int(return_dict['nose'][1])), 15, (0,255,0), -1)
    cv2.circle(img_copy, (int(return_dict['hip'][0]), int(return_dict['hip'][1])), 15, (0,255,0), -1)
    
    if return_dict['direction'] == 'right':
        sign = 1
    else:
        sign = -1
    
    # Drawing the 2 lines for lean
    cv2.line(img_copy, (int(return_dict['nose'][0]), int(return_dict['nose'][1])), (int(return_dict['hip'][0]), int(return_dict['hip'][1])), (0,255,0), 10)
    cv2.line(img_copy, (int(return_dict['nose'][0]+50*sign), int(return_dict['hip'][1])), (int(return_dict['hip'][0]), int(return_dict['hip'][1])), (0,255,0), 10)
    
    # Circle for angle
    cv2.circle(img_copy, (int(return_dict['hip'][0]), int(return_dict['hip'][1])), 50, (0,255,0), 5)
    
    # Angle between hip and nose
    radians2 = math.atan2(return_dict['nose'][1] - return_dict['hip'][1], return_dict['nose'][0] - return_dict['hip'][0])
    degrees2 = math.degrees(radians2)
    if return_dict['direction'] == "left":
        degrees2 = (degrees2 + 180)*-1
    degrees_str2 = str(round(degrees2*-1, 2))
    return_dict['lean'] = round(degrees2*-1, 2)
    
    if return_dict['direction'] == "left":
        x = -150
    else:
        x = 50
    # Label the angle
    cv2.putText(img_copy, degrees_str2, (int(return_dict['hip'][0])+x, int(return_dict['hip'][1])-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
    
    # Drawing the 2 lines for stride
    cv2.line(img_copy, (int(return_dict['ankle'][0]), int(return_dict['ankle'][1])), (int(return_dict['knee'][0]), int(return_dict['knee'][1])), (255,0,0), 10)
    cv2.line(img_copy, (int(return_dict['ankle'][0]+100*sign), int(return_dict['knee'][1])), (int(return_dict['knee'][0]), int(return_dict['knee'][1])), (255,0,0), 10)
    
    # Circle for angle
    cv2.circle(img_copy, (int(return_dict['knee'][0]), int(return_dict['knee'][1])), 50, (255,0,0), 5)
    
    # Angle between knee and ankle
    radians = math.atan2(return_dict['knee'][1] - return_dict['ankle'][1], return_dict['knee'][0] - return_dict['ankle'][0])
    degrees = math.degrees(radians)
    if return_dict['direction'] == "right":
        degrees_str = str(round(180+degrees, 2))
        x1 = 50
        return_dict['stride'] = round(180+degrees, 2)
    else:
        degrees_str = str(round(degrees*-1, 2))
        x1 = -150
        return_dict['stride'] = round(degrees*-1, 2)
    
    # Label the angle
    cv2.putText(img_copy, degrees_str, (int(return_dict['knee'][0])+x1, int(return_dict['knee'][1])+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3, cv2.LINE_AA)
    
    # Update the return dict
    return_dict['image'] = img_copy
    
    return return_dict