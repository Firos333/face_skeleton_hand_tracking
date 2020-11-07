import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


######
import mediapipe as mp


#####

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
fps_time = 0
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('output.avi',fourcc, 1.0, (640,480))

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    # For webcam input:
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

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
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read()
        w, h = model_wh(args.resolution)
        humans = e.inference(image,resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        c=1
        for (x,y,w,h) in faces: 
        	cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),2) 
        	cv2.putText(image, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        	c+=1
        ims = cv2.resize(image, (980, 540)) 
        image = cv2.cvtColor(ims, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #frame = cv2.flip(ims,0)
        out.write(ims)
        out.release()
    cv2.destroyAllWindows()
    
logger.debug('finished+')
