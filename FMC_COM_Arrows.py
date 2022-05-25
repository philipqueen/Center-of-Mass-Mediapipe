import cv2
import mediapipe as mp
import numpy as np
import math
import pickle
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

### center of mass functions - eventually we would make these a class
def calculateCOM(proximal, distal, com_proximal_multiplier):
    segment_length = distal-proximal
    segment_COM = proximal + (com_proximal_multiplier*segment_length)
    return segment_COM

def calculate_foot_COM(heel, ankle, foot_index):
    COM_foot = (heel + ankle + foot_index)/3
    return COM_foot

def calculate_hand_COM(wrist, index, pinky, com_proximal_multiplier):
    knuckle_width = pinky - index
    Third_metacarple = index + (knuckle_width/3)
    Palm_segment = Third_metacarple - wrist
    COM_hand = wrist + (com_proximal_multiplier*Palm_segment)
    return COM_hand

def calculate_total_COM(COM_Segments, dimension):
    '''Calculate total body COM based on COM of all segments, separately for x and y. 
    Dimension should be 0 for x and 1 for y.'''

    return sum([COM_Segments[segment][2]*COM_Segments[segment][dimension] for segment in COM_Segments])

def handCOM(landmarks):
    # COM of Hands, creating third metacarple point by taking 1/3 of the distance of the segment between index and pinky landmarks
    # then adding that distance to the index. Palm segment will be from wrist joint to 3rd metacarple joint.
    # dictionary values arranged in [wrist, index, pinky, com_proximal_multiplier, com_x, com_y] in that order. 
    Hands = {
        'L_hand' : [landmarks[15], landmarks[19], landmarks[17], 0.7474, 0, 0],
        'R_hand' : [landmarks[16],landmarks[20], landmarks[18], 0.7474, 0, 0],
    }

    for key in Hands:
        #Knuckle_width is defined as the segment from the distal ends of the second to fifth metacarples (index to pinky landmarks)
        wrist_x = Hands[key][0][0]
        wrist_y = Hands[key][0][1]
        index_x = Hands[key][1][0]
        index_y = Hands[key][1][1]
        pinky_x = Hands[key][2][0]
        pinky_y = Hands[key][2][1]
        com_proximal_multiplier = Hands[key][3] 

        COM_hand_x = calculate_hand_COM(wrist_x, index_x, pinky_x, com_proximal_multiplier)
        COM_hand_y = calculate_hand_COM(wrist_y, index_y, pinky_y, com_proximal_multiplier)

        #update Hands dictionary with COM positions
        Hands[key][4] = COM_hand_x
        Hands[key][5] = COM_hand_y
    return Hands

def feetCOM(landmarks):
    #COM of FEET, using the centroid of the triangle formed from heel-ankle-"foot-index" landmarks, arranged in that order in the dictionary values.
    Feet = {
        'L_foot' : [landmarks[29], landmarks[27], landmarks[31], 0, 0],
        'R_foot' : [landmarks[30], landmarks[28], landmarks[32], 0, 0]
    }

    for key in Feet:
        #arguments are ordered heel, ankle, foot index
        COM_foot_x = calculate_foot_COM(Feet[key][0][0], Feet[key][1][0], Feet[key][2][0])
        COM_foot_y = calculate_foot_COM(Feet[key][0][1], Feet[key][1][1], Feet[key][2][1])

        #update Feet dictionary with COM positions
        Feet[key][3] = COM_foot_x
        Feet[key][4] = COM_foot_y
        
    return Feet

def trunkCOM(landmarks):
    Trunk = []
    
    #Trunk segment calculations from MidShoulder to Mid Hip. 
    MidShoulder_x = (landmarks[12][0] + landmarks[11][0])/2
    MidShoulder_y = (landmarks[12][1] + landmarks[11][1])/2
    MidHip_x = (landmarks[24][0] + landmarks[23][0])/2
    MidHip_y = (landmarks[24][1] + landmarks[23][1])/2
    TrunkCOM_x = calculateCOM(MidShoulder_x, MidHip_x, 0.3782)
    TrunkCOM_y = calculateCOM(MidShoulder_y, MidHip_y, 0.3782)
    
    Trunk.append(TrunkCOM_x)
    Trunk.append(TrunkCOM_y)
    
    return Trunk

def body_segmentsCOM(landmarks):
    #Body Segment Dictionary format: key = body segment, % value = [proximal joint landmark values, distal joint landmark values, COM as a % of segment length]
    Body_Segments = {
        'L_UpperArm' : [landmarks[11] , landmarks[13] , 0.5754, 0, 0],
        'R_UpperArm' : [landmarks[12] , landmarks[14] , 0.5754, 0, 0], 
        'L_Forearm' : [landmarks[13] , landmarks[15] , 0.4559, 0, 0],
        'R_Forearm' : [landmarks[14] , landmarks[16] , 0.4559, 0, 0], 
        'L_Thigh' : [landmarks[23], landmarks[25], 0.3612, 0, 0], 
        'R_Thigh' : [landmarks[24], landmarks[26], 0.3612, 0, 0],
        'L_Shin' : [landmarks[25], landmarks[27], 0.4416, 0, 0],
        'R_Shin' : [landmarks[26], landmarks[28], 0.4416, 0, 0],
        'Head' : [landmarks[7], landmarks[8], 0.5, 0, 0]
    }


    for key in Body_Segments:
        #print(key, 'proximal x ->', Body_Segments[key][0].x) 
        x1 = Body_Segments[key][0][0] #proximal joint x value
        y1 = Body_Segments[key][0][1]
        x2 = Body_Segments[key][1][0]
        y2 = Body_Segments[key][1][1]
        com_proximal_multiplier = Body_Segments[key][2]

        COM_x = calculateCOM(x1, x2, com_proximal_multiplier)
        COM_y = calculateCOM(y1, y2, com_proximal_multiplier)
        
        Body_Segments[key][3] = COM_x
        Body_Segments[key][4] = COM_y
        
    return Body_Segments

def define_COM_segments(Body_Segments, Trunk, Feet, Hands):
    COM_Segments = {
                'Head' : [ Body_Segments['Head'][3], Body_Segments['Head'][4], 0.0668],
                'Trunk' : [ Trunk[0], Trunk[1], 0.4257 ],
                'Left_Upper_Arm' : [ Body_Segments['L_UpperArm'][3], Body_Segments['L_UpperArm'][4], 0.0255],
                'Right_Upper_Arm' : [ Body_Segments['R_UpperArm'][3], Body_Segments['R_UpperArm'][4], 0.0255],
                'Left_Forearm' : [ Body_Segments['L_Forearm'][3], Body_Segments['L_Forearm'][4], 0.0138],
                'Right_Forearm' : [ Body_Segments['R_Forearm'][3], Body_Segments['R_Forearm'][4], 0.0138],
                'Left_Hand' : [Hands['L_hand'][4], Hands['L_hand'][5], 0.0056],
                'Right_Hand' : [Hands['R_hand'][4], Hands['R_hand'][5], 0.0056],
                'Left_Thigh' : [Body_Segments['L_Thigh'][3], Body_Segments['L_Thigh'][4], 0.1478],
                'Right_Thigh' : [Body_Segments['R_Thigh'][3], Body_Segments['R_Thigh'][4], 0.1478],
                'Left_Shin' : [Body_Segments['L_Shin'][3], Body_Segments['L_Shin'][4], 0.0481],
                'Right_Shin' : [Body_Segments['R_Shin'][3], Body_Segments['R_Shin'][4], 0.0481],
                'Left_Foot' : [Feet['L_foot'][3], Feet['L_foot'][4], 0.0129], 
                'Right_Foot' : [Feet['R_foot'][3], Feet['R_foot'][4], 0.0129], 
            }
    return COM_Segments

def get_total_COM(COM_Segments):
    COM_total_x = calculate_total_COM(COM_Segments, 0) #0 points to x dimension
    COM_total_y = calculate_total_COM(COM_Segments, 1) #1 points to y dimension
    
    return [COM_total_x, COM_total_y]

def update_COM_Segments(COM_Segments, total_COM):
    COM_Segments['Total'] = total_COM
    
def get_COM_dict(landmarks):
    
    Hands = handCOM(landmarks)
    Feet = feetCOM(landmarks)
    Trunk = trunkCOM(landmarks)
    Body_Segments = body_segmentsCOM(landmarks)
    COM_Segments = define_COM_segments(Body_Segments, Trunk, Feet, Hands)
    
    total_COM = get_total_COM(COM_Segments)
    
    update_COM_Segments(COM_Segments, total_COM)

    
    return COM_Segments

def normalize_landmarks(landmarks, width, height):
    #this doesn't give the same values as the naive script, need to investigate more
    for landmark in landmarks:
        landmark[0] /= width
        landmark[1] /= height

    return landmarks


def get_whole_video_COM(mediaPipeData, camera, video_path):
    print("calculating COM...")
    whole_video_COM = []

    #open video capture to get width and height properties
    cap = cv2.VideoCapture(video_path)

    #get width and height properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    for frame in mediaPipeData[camera]:
        try:
            landmarks = normalize_landmarks(frame,width, height)
            COM_dict = get_COM_dict(landmarks)
            whole_video_COM.append(COM_dict)
        except:
            #if pose isn't detected, we fill out dictionary with NotANumber values
            nan_dict = {'Head': [math.nan, math.nan, math.nan],
            'Trunk': [math.nan, math.nan, math.nan],
            'Left_Upper_Arm': [math.nan, math.nan, math.nan],
            'Right_Upper_Arm': [math.nan, math.nan, math.nan],
            'Left_Forearm': [math.nan, math.nan, math.nan],
            'Right_Forearm': [math.nan, math.nan, math.nan],
            'Left_Hand': [math.nan, math.nan, math.nan],
            'Right_Hand': [math.nan, math.nan, math.nan],
            'Left_Thigh': [math.nan, math.nan, math.nan],
            'Right_Thigh': [math.nan, math.nan, math.nan],
            'Left_Shin': [math.nan, math.nan, math.nan],
            'Right_Shin': [math.nan, math.nan, math.nan],
            'Left_Foot': [math.nan, math.nan, math.nan],
            'Right_Foot': [math.nan, math.nan, math.nan],
            'Total': [math.nan, math.nan]}
            whole_video_COM.append(nan_dict)

    return whole_video_COM

def get_COM_values(whole_video_COM, key, index):
    '''Gets the total body COM value and returns it as a numpy array.
    Set index to 0 for X values and 1 for Y values.'''

    total = [frame[key][index] for frame in whole_video_COM]

    np_total = np.asarray(total)
    return np_total

def pad_array(array):
    '''Adds a 0 to the beginning of a numpy array.'''

    array = np.concatenate(([0],array)) #concatenate our array to the [0] array
    #array = np.append(array, 0) #this adds a 0 to the end, 

    return array

def get_rate_data(whole_video_COM, key):
    #this gets total COM data out of our list of dictionaries
    x = get_COM_values(whole_video_COM, key, 0)
    y = get_COM_values(whole_video_COM, key, 1)

    #calculate velocities
    velocity_x = np.diff(pad_array(x)) #use padded array to make sure arrays stay the same length
    velocity_y = np.diff(pad_array(y))

    #calculate accelerations
    accel_x = np.diff(pad_array(velocity_x)) #use padded array to make sure arrays stay the same length
    accel_y = np.diff(pad_array(velocity_y))

    #store total COM total, velocity, and accelerations in list for easy passing to functions
    rate_data = [x, y, velocity_x, velocity_y, accel_x, accel_y]

    return rate_data

def mediapipe_video(path):
    #capture the video using opencv
    cap = cv2.VideoCapture(path)

    results_array = [] #this will store pose estimations from mediapipe

    print("getting pose estimation...")
    with mp_pose.Pose(model_complexity = 2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): #loop through feed
            ret, frame = cap.read() #getting an image from feed, 'frame' is our video feed variable
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #recolor image into the RGB format (for mediapipe)
            image.flags.writeable = False
            
            #make pose detection in MediaPipe
            results = pose.process(image)
        
            results_array.append(results)

            if cv2.waitKey(10) & 0xFF == ord('q'): #break out of feed by typing 'q' key
                break

    cap.release()

    #numpy-ify array
    np_results = np.asarray(results_array)

    return np_results

def display_COM(whole_video_COM, frame_idx, frame, width, height):
    for key in whole_video_COM[frame_idx]:
        try: #need try to handle Nan values where pose it not detected
            if key == 'Total':
                cv2.circle(frame, center=tuple(np.multiply((whole_video_COM[frame_idx][key][0], whole_video_COM[frame_idx][key][1]), [width, height]).astype(int)), radius=2, color=(0,255,0), thickness=8)
            else:
                cv2.circle(frame, center=tuple(np.multiply((whole_video_COM[frame_idx][key][0], whole_video_COM[frame_idx][key][1]), [width, height]).astype(int)), radius=2, color=(255,0,0), thickness=4)
        
        except:
            pass

def display_arrow(data_list, derivative_order, scale_factor, color, frame_idx, frame, width, height):
    #calculate data_list index based on derivative order (1 for velocity, 2 for acceleration)
    deriv_x = derivative_order * 2
    deriv_y = (derivative_order * 2) + 1 

    #set start and end point for arrow (scale factor determines magnitude of drawn arrow)
    arrow_start = tuple(np.multiply((data_list[0][frame_idx], data_list[1][frame_idx]), [width, height]).astype(int)) #start point if just the COM
    arrow_end = tuple(np.add(arrow_start, np.multiply((data_list[deriv_x][frame_idx], data_list[deriv_y][frame_idx]), scale_factor)).astype(int)) #end point is start point with velocity times arrow size added to it

    #draw the arrow
    cv2.arrowedLine(frame, arrow_start, arrow_end, color, thickness = 8)

def display_final_video(path, whole_video_COM, rate_data1, rate_data2, rate_data3, rate_data4, rate_data5):
    #reset capture
    cap = cv2.VideoCapture(path)

    #setup properties for video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(width, height)

    #create path for saving video output
    output_path = path.split(".")[0] + "_COM" + ".mp4"

    #create video writer
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("displaying and writing video...")
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): #loop through feed
        ret, frame = cap.read() #getting an image from feed, 'frame' is our video feed variable

        #display COM dots
        display_COM(whole_video_COM, frame_idx, frame, width, height)   

         #display velocity arrows
        try: #handle NaNs where COM isn't tracked
            display_arrow(rate_data1, 1, 20000, (0,0,128), frame_idx, frame, width, height)
            display_arrow(rate_data2, 1, 6000, (0,128,0), frame_idx, frame, width, height)
            display_arrow(rate_data3, 1, 6000, (0,128,0), frame_idx, frame, width, height)
            display_arrow(rate_data4, 1, 6000, (0,128,0), frame_idx, frame, width, height)
            display_arrow(rate_data5, 1, 6000, (0,128,0), frame_idx, frame, width, height)
        except:
            pass

        #display image - comment out to just save video
        cv2.imshow('COM Display (q to quit)', frame)

        #write video frame to file - comment out to just view video
        writer.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'): #break out of feed by typing 'q' key
            break

            

    #release camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1) #helps close windows for certain mac users


    #release video writer
    writer.release()

##########################################################################################################################
########################################################################################################################## 
########################################################################################################################## 
########################################################################################################################## 
########################################################################################################################## 

#eventually this would be def main()

#set session folder path
session_folder_path = "/Users/Philip/Documents/Humon Research Lab/fmc_COM/philip_session2_04_25_22"

#get data path as the mediaPipeData_2d.npy file location
data_path = session_folder_path + "/DataArrays/mediaPipeData_2d.npy"

#pick camera
camera = 0

#get video path
video_path = session_folder_path + "/SyncedVideos/synced_Cam" + str(camera) + ".MP4"

#load in saved mediapipe data
mediaPipeData = np.load(data_path)

#get whole_video_COM dictionary our of mediapipe data
whole_video_COM = get_whole_video_COM(mediaPipeData, camera, video_path)
print(whole_video_COM[0])

#rate data holds position, velocity, and acceleration data for a given segment
total_rate_data = get_rate_data(whole_video_COM, 'Total')
lhand_rate_data = get_rate_data(whole_video_COM, 'Left_Hand')
rhand_rate_data = get_rate_data(whole_video_COM, 'Right_Hand')
lfoot_rate_data = get_rate_data(whole_video_COM, 'Left_Foot')
rfoot_rate_data = get_rate_data(whole_video_COM, 'Right_Foot')
print(total_rate_data)
#save total body rate data as .npy file
total_rate_npy = np.asarray(total_rate_data)
print(total_rate_npy)
npy_save_path = session_folder_path + "/DataArrays/" + "TotalBodyCOMdata.npy"
np.save(npy_save_path, total_rate_data)

#display and write video with COM points and velocity arrows
#display_final_video(video_path, whole_video_COM, total_rate_data, lhand_rate_data, rhand_rate_data, lfoot_rate_data, rfoot_rate_data)