import argparse
import logging
import time
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from mtcnn.mtcnn import MTCNN
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
#q = queue.Queue()
#import ctypes 

class myThread(threading.Thread):

    def __init__(self, image, name,q):
        threading.Thread.__init__(self)
        self.image = image
        print(image,name,q)
        self.name = name
        self.q = q
    def run(self):
        #result = detector.detect_faces(self.image)
        result=self.image
        w, h = model_wh(args.resolution)
        humans = e.inference(self.image,resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        if not args.showBG:
            image = np.zeros(self.image.shape)
        
        image = cv2.resize(self.image, (980, 540)) 
        image0 = image[0:270, 0:245]
        image1 = image[0:270, 245:490]
        image2 = image[0:270, 490:735]
        image3 = image[0:270, 735:980]
        image4 = image[270:540, 0:245]
        image5 = image[270:540, 245:490]
        image6 = image[270:540, 490:735]
        image7 = image[270:540, 735:980]
        img1 = [image0,image1,image2,image3,image4,image5,image6,image7]
        result_list =[]
        for image in img1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result_list.append(results)
		#q.task_done()
		#return result,humans,result_list
        #workQueue.put(result,humans,result_list)
        workQueue.put(result) #rectangular box dimensions of face
        workQueue.put(humans) # coordinates of pose
        workQueue.put(result_list) # a list of handmarks for 8 cropped image (x,y,z c00rdinates)
        
                      
            
def mark(image,result,humans,result_list):
    
    if result==[]:
        return(image)
    
    image = cv2.resize(image, (980, 540)) 
    image0 = image[0:270, 0:245]
    image1 = image[0:270, 245:490]
    image2 = image[0:270, 490:735]
    image3 = image[0:270, 735:980]
    image4 = image[270:540, 0:245]
    image5 = image[270:540, 245:490]
    image6 = image[270:540, 490:735]
    image7 = image[270:540, 735:980]
    
    img1 = [image0,image1,image2,image3,image4,image5,image6,image7]
    img =[]
    zipp=zip(img1,result_list)
    for image,results in zipp:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        img.append(image)
    im_half1 = cv2.hconcat([img[0], img[1],img[2],img[3]])
    im_half2 = cv2.hconcat([img[4], img[5],img[6],img[7]])
    image = cv2.vconcat([im_half1, im_half2])
        
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          
    return(image)
  
if __name__ == '__main__':

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
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
    
    global w,h,i
    
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (980, 540)
    result1 = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         29, size) 
    
    
    #cap.set(cv2.CV_CAP_PROP_FPS, 10) 
    detector = MTCNN()
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    ctn=0
    global i
    i=0
    #workQueue = Queue.Queue(1)
    #th = threading.Thread(target=processs, args=(),daemon=True)
    #th.start()
    exitFlag = 0
    queueLock = threading.Lock()
    workQueue = queue.Queue(3)
    
    result_list =[]
    result=[]
    humans=[]
    while cap.isOpened():
        ret_val, image = cap.read()
        ctn+=1
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        if ctn==1:
            #item=image
            #q.put(item)
            th= myThread(image, ctn, workQueue)
            th.start()
            #th.join()
            print(th)
            #result,humans,result_list = q.put(item)
        if workQueue.qsize()==3:
            #th.join()
            #th.stop()
            #result,humans,result_list = workQueue.get()
            result =workQueue.get()
            humans =workQueue.get()
            result_list =workQueue.get()
            print(result,humans,result_list)
            th= myThread(image, ctn, workQueue)
            th.start()
        
        image=mark(image,result,humans,result_list)
        print(result,humans,result_list)
        
 
        image = cv2.resize(image, (980, 540)) 
        cv2.imshow('tf-pose-estimation result', image)
        result1.write(image)
        fps_time = time.time()
        
        if cv2.waitKey(1) == 27:
            result1.release()
            print("The video was successfully saved") 
            break
    
    
    #exitFlag = 1
    result1.release() 
    cv2.destroyAllWindows()
    print("The video was successfully saved") 
    
    
logger.debug('finished+')
