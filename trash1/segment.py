import argparse

import cv2
import numpy as np

import pixellib
from pixellib.tune_bg import alter_bg

#####

change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
change_bg.blur_video("me6.mp4",extreme=True,frames_per_second=10,output_video_name="blur_video.mp4")

#change_bg.color_bg("sample.jpg", colors = (0,128,0), output_image_name="output_img.jpg", detect = "person")
