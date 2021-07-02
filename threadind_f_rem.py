import argparse
import logging
import time
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import mediapipe as mp

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
fps_time = 0   
import threading, queue
import re
#q = queue.Queue()
#import ctypes 

#make a thread for doing process face,hand and pose it will return the landmarks and dimentions for marking the portions
class myThread1(threading.Thread):

    def __init__(self, image, frame_no ,q):
        threading.Thread.__init__(self)
        self.image = image
        self.frame_no = frame_no
        self.q = q
    def run(self):
        
        
        image_sized = cv2.resize(self.image, (980, 540))   #resize image
        
        #crop the sized image to 8 parts
        
        img_cropped =[]
        height, width, channels = image_sized.shape
        # Number of pieces Horizontally 
        CROP_W_SIZE  = 4 
        # Number of pieces Vertically to each Horizontal  
        CROP_H_SIZE = 2 
        for ih in range(CROP_H_SIZE):
            for iw in range(CROP_W_SIZE):
                x = int((width/CROP_W_SIZE) * iw )
                y = int((height/CROP_H_SIZE) * ih)
                h = int((height / CROP_H_SIZE))
                w = int((width / CROP_W_SIZE ))
                img = image_sized[y:y+h, x:x+w]
                #np.append(img_cropped,img)
                img_cropped.append(img)
        hand_result =[]
        for image in img_cropped:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)    #process hand detection for frame  .. this for loop will do  the same for cropped images
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            hand_result.append(results)
            
        workQueue.put(hand_result) # a list of handmarks for 8 cropped image (x,y,z c00rdinates)
        

class myThread2(threading.Thread):

    def __init__(self, image, frame_no ,q):
        threading.Thread.__init__(self)
        self.image = image
        self.frame_no = frame_no
        self.q = q
    def run(self):
        
        w, h = model_wh(args.resolution)
        #send frame to do pose detection
        pose_result = e.inference(self.image,resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        if not args.showBG:
            image = np.zeros(self.image.shape)        
        workQueue.put(pose_result) # coordinates of pose

                   
#function for mark the face,hand and pose in the current frame          
def mark(image,pose_result,hand_result):
    
    if pose_result==[]:
        return(image)

    w, h = model_wh(args.resolution)    #resize image for pose detection
    image = cv2.resize(image, (w,h))
    #print(pose_result)
    for pose in pose_result:
        string_pose = str(pose)
        print(string_pose)
        
        if 'BodyPart:2' in string_pose and  'BodyPart:5' in string_pose:
            x_str = re.findall(r"BodyPart:2-(.*?),", string_pose) #convert landmarks out to string and take x values
            print(x_str)
            x =float(x_str[0].replace("(", ""))
            y_str = re.findall(r"BodyPart:5-(.*?),", string_pose)
            y =float(y_str[0].replace("(", ""))
            
            face = string_pose[12:22].replace(" ", "")  # take nose values
            face= face.split(', ')
            face_points= face[0].split(',')
            x_f = int(float(face_points[0])*w)  # value of width of nose postion
            y_f = int(float(face_points[1])*h)  #value of hieght of nose postion
            left_x = int(x*980)
            right_x = int(y*980)
            #print(left_x,right_x)
            dif= int((right_x-left_x)*0.15)  # find shoulder length and multiply with propotionality constant
            #draw rectangle on face nose as centre
            cv2.rectangle(image, (x_f-dif,y_f-dif-10), (x_f+dif,y_f+dif+10), (0,0,0), 1)
        
    
    image_sized = cv2.resize(image, (980, 540)) 
    img_cropped =[]
    height, width, channels = image_sized.shape
    # Number of pieces Horizontally 
    CROP_W_SIZE  = 4 
    # Number of pieces Vertically to each Horizontal  
    CROP_H_SIZE = 2 
    for ih in range(CROP_H_SIZE):
        for iw in range(CROP_W_SIZE):
            x = int((width/CROP_W_SIZE) * iw )
            y = int((height/CROP_H_SIZE) * ih)
            h = int((height / CROP_H_SIZE))
            w = int((width / CROP_W_SIZE ))
            img = image_sized[y:y+h, x:x+w]
            #np.append(img_cropped,img)
            img_cropped.append(img)

    zipp=zip(img_cropped,hand_result)
    for image,results in zipp:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    im_half1 = cv2.hconcat([img_cropped[0], img_cropped[1],img_cropped[2],img_cropped[3]])  # join horizontally cropped image 
    im_half2 = cv2.hconcat([img_cropped[4], img_cropped[5],img_cropped[6],img_cropped[7]])  # join horizontally cropped image 
   
    image_joined = cv2.vconcat([im_half1, im_half2])               # combine both horizontal to one image
        
    image_marked = TfPoseEstimator.draw_humans(image_joined,pose_result, imgcopy=False)  # draw pose estimation
    #put a text on image to show fps
    cv2.putText(image_marked, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          
    return(image_marked)
  
if __name__ == '__main__':

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands  #define hands
    hands = mp_hands.Hands(max_num_hands=5,min_detection_confidence=0.5, min_tracking_confidence=0.4)
    #static_image_mode=True,min_detection_confidence=0.5,max_num_hands=12,,min_tracking_confidence=0.5
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    
    global w,h
    
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #fix the size of the out put video file
    size = (980, 540)
    result1 = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         2, size) 
       
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    frame_no =0 # this will give the nth frame's number

    queueLock = threading.Lock()
    workQueue = queue.Queue(2)  # set maximum queue size as 3
    
    hand_result,face_result,pose_result =([], ) * 3  

    while cap.isOpened():
        ret_val, image = cap.read()   #read frame
        frame_no+=1  # increment number of frame
        
        image = cv2.rotate(image, cv2.cv2.ROTATE_180)
        if frame_no ==1:

            t1= myThread1(image,frame_no, workQueue)  # initialise thread to do first frame
            t1.start()  
            t2= myThread2(image,frame_no, workQueue) 
            t2.start()  

        if workQueue.qsize()==2:   # if the queue size is three
            
            hand_result =workQueue.get()
            pose_result =workQueue.get()

            
            t1= myThread1(image,frame_no, workQueue)  # initialise thread to do first frame
            t1.start()  
            t2= myThread2(image,frame_no, workQueue)  # initialise thread to do first frame
            t2.start()  
 
        
        image_with_marks =mark(image,pose_result,hand_result)   
   
        #resize frame to show on cv window
        image_resized = cv2.resize(image_with_marks, (980, 540)) 
        cv2.imshow('Face+Hand+Pose result', image_resized)
        #write image to output file
        result1.write(image_resized)
        fps_time = time.time()
        
        if cv2.waitKey(1) == 27:
            result1.release()   #if Esc is pressed save the outputfile as video
            print("The video was successfully saved") 
            break
    
    
    #if frames completed is pressed save the outputfile as video
    result1.release() 
    cv2.destroyAllWindows()
    print("The video was successfully saved") 
    
    
logger.debug('finished+')

