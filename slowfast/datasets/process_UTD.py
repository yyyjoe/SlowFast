from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

import os
import os.path
import csv
import numpy as np
import sys

def generate_subclip(dirName,output_dirName):

    listOfFile = os.listdir(dirName)
    
    for f in listOfFile:
        video_name = os.path.join(dirName,f)
        if os.path.isdir(video_name):
            continue
        clip = VideoFileClip(video_name)
        duration = clip.duration
        print("Duration of video : ", clip.duration)
        print("FPS : ", clip.fps)
        target_name = os.path.join(output_dirName,f)
        myclip2 = clip.subclip(0.5, duration-0.3)
        myclip2.write_videofile(target_name, codec = "libx264", fps=clip.fps)
        myclip2.close()


data_path = "/scratch/user/yiminchou1994/UTD/RGB"
generate_subclip(data_path,os.path.join(data_path,"short"))
