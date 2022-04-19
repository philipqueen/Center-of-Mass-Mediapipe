import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

### center of mass functions - eventually we would make these a class
#change these to take in tuples (proximal_x, proximal_y), (distal_x, distal_y)
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

#### 

#eventually this would be def main()

#set file path to the video location
path = "/Users/Philip/Documents/GitHub/Learning/Videos/delaney_almighty.mp4"

#capture the video using opencv
cap = cv2.VideoCapture(path)

#setup properties for video writer
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width_height = (int(width), int(height))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = "/Users/Philip/Documents/GitHub/Learning/Videos/delaney_almighty_COM.mp4"

#create video writer
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)


with mp_pose.Pose(model_complexity = 2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): #loop through feed
        ret, frame = cap.read() #getting an image from feed, 'frame' is our video feed variable
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #recolor image into the RGB format (for mediapipe)
        image.flags.writeable = False
        
        #make detection, accessing our pose variable. processing the pose 
        #..variable to get our detections and then storing those detections
        #..into the 'results' variable
        results = pose.process(image)
        image.flags.writeable = True #setting this to true allows the drawing of the landmarks onto the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #recolor image back to BGR (for opencv)

        #### NEW: EXTRACT LANDMARKS####
        try: #allows for the random dropped frames in webcam video feeds
            landmarks = results.pose_landmarks.landmark
            
            # COM of Hands, creating third metacarple point by taking 1/3 of the disatance of the segment between index and pinky landmarks
            # then adding that distance to the index. Palm segment will be from wrist joint to 3rd metacarple joint.
            # dictionary values arranged in [wrist, index, pinky, com_proximal_multiplier, com_x, com_y] in that order. 
            Hands = {
                'L_hand' : [landmarks[15], landmarks[19], landmarks[17], 0.7474, 0, 0],
                'R_hand' : [landmarks[16],landmarks[20], landmarks[18], 0.7474, 0, 0],
            }
          
            for key in Hands:
                #Knuckle_width is defined as the segment from the distal ends of the second to fifth metacarples (index to pinky landmarks)
                wrist_x = Hands[key][0].x
                wrist_y = Hands[key][0].y
                index_x = Hands[key][1].x
                index_y = Hands[key][1].y
                pinky_x = Hands[key][2].x
                pinky_y = Hands[key][2].y
                com_proximal_multiplier = Hands[key][3] 

                COM_hand_x = calculate_hand_COM(wrist_x, index_x, pinky_x, com_proximal_multiplier)
                COM_hand_y = calculate_hand_COM(wrist_y, index_y, pinky_y, com_proximal_multiplier)

                cv2.circle(image, center=tuple(np.multiply((COM_hand_x, COM_hand_y), [width, height]).astype(int)), radius=1, color=(255,0,0), thickness=2)
                
                #update Hands dictionary with COM positions
                Hands[key][4] = COM_hand_x
                Hands[key][5] = COM_hand_y

            # print('update',  Hands[key][4], Hands[key][5]) # YAY, now the fourth thing in the list for each key is changed to COM_hand_x and the 5th thing to COM_hand_y each time it iterates. 
            # print("original", Hands)

            #COM of FEET, using the centroid of the triangle formed from heel-ankle-"foot-index" landmarks, arranged in that order in the dictionary values.
            Feet = {
                'L_foot' : [landmarks[29], landmarks[27], landmarks[31], 0, 0],
                'R_foot' : [landmarks[30], landmarks[28], landmarks[32], 0, 0]
            }

            for key in Feet:
                #arguments are ordered heel, ankle, foot index
                COM_foot_x = calculate_foot_COM(Feet[key][0].x, Feet[key][1].x, Feet[key][2].x)
                COM_foot_y = calculate_foot_COM(Feet[key][0].y, Feet[key][1].y, Feet[key][2].y)

                #plot feet COM
                cv2.circle(image, center=tuple(np.multiply((COM_foot_x, COM_foot_y), [width, height]).astype(int)), radius=1, color=(255,0,0), thickness=2)

                #update Feet dictionary with COM positions
                Feet[key][3] = COM_foot_x
                Feet[key][4] = COM_foot_y

            
            #Trunk segment calculations from MidShoulder to Mid Hip. 
            MidShoulder_x = (landmarks[12].x + landmarks[11].x)/2
            MidShoulder_y = (landmarks[12].y + landmarks[11].y)/2
            MidHip_x = (landmarks[24].x + landmarks[23].x)/2
            MidHip_y = (landmarks[24].y + landmarks[23].y)/2
            TrunkCOM_x = calculateCOM(MidShoulder_x, MidHip_x, 0.3782)
            TrunkCOM_y = calculateCOM(MidShoulder_y, MidHip_y, 0.3782)

            cv2.circle(image, center=tuple(np.multiply((TrunkCOM_x, TrunkCOM_y), [width, height]).astype(int)), radius=1, color=(255,0,0), thickness=2)
            # cv2.circle(image, center=tuple((TrunkCOM_x*width, TrunkCOM_y*height)), radius=4, color=(255,0,0), thickness=2)

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
                x1 = Body_Segments[key][0].x #proximal joint x value
                y1 = Body_Segments[key][0].y
                x2 = Body_Segments[key][1].x
                y2 = Body_Segments[key][1].y
                com_proximal_multiplier = Body_Segments[key][2]

                COM_x = calculateCOM(x1, x2, com_proximal_multiplier)
                COM_y = calculateCOM(y1, y2, com_proximal_multiplier)
                #print('com_x =', COM_x)
                #print('com_y =', COM_y)
            
                #Render COM_x and COM_y of the 8 limb segments onto the video feed. 
                cv2.circle(image, center=tuple(np.multiply((COM_x, COM_y), [width, height]).astype(int)), radius=1, color=(255,0,0), thickness=2)

                Body_Segments[key][3] = COM_x
                Body_Segments[key][4] = COM_y

            COM_Segments = {
                'Head' : [ Body_Segments['Head'][3], Body_Segments['Head'][4], 0.0668],
                'Trunk' : [ TrunkCOM_x, TrunkCOM_y, 0.4257 ],
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

            COM_total_x = calculate_total_COM(COM_Segments, 0) #0 points to x dimension
            COM_total_y = calculate_total_COM(COM_Segments, 1) #1 points to y dimension
                

            cv2.circle(image, center=tuple(np.multiply((COM_total_x, COM_total_y), [width, height]).astype(int)), radius=2, color=(0,255,0), thickness=5)
                # So, I think I need to have the for loop give me the percent multiplier but use something along the lines of:
                # values = dictionary.values()
                # total = sum(values)
                # the currrent for loop calculation isn't summing anything together. 

        except:
            pass

        # Render Detections ... 'results.pose_landmarks' delivers the coordinantes of the landmarks.
        #... 'mp_pose.POSE_CONNECTIONS' tells you whats connected to what (shoulder - elbow for example)
        # I commeted out the line below (skeleton landmarks and segments) because the video was getting a little crowded and I couldn't see the head COM behind the cluster of face landmarks. 

        '''mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,128), thickness=2, circle_radius=2), #set color for joints in BGR
                                mp_drawing.DrawingSpec(color=(0,128,128), thickness=2, circle_radius=2) #set color for connections in BGR
                                )'''

        cv2.imshow('COM Skeleton', image) #will allow us to visualize image with the landmarks drawn

        #write video frame to file
        writer.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'): #break out of feed by typing 'q' key
            break

    cap.release()
    cv2.destroyAllWindows() #will close any window open with your image
    cv2.waitKey(1) #helps close windows for certain mac users

    #release video writer
    writer.release()

    