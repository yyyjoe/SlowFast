from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

import os
import os.path
import csv
import numpy as np
import sys
import argparse
from UTD.download import download_avi_files

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='foo help')
args = parser.parse_args()
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

def load_video_generate_csv(dirName):
    file_names = os.listdir(dirName)
    train_set = []
    val_set = []
    for f in file_names:
        split = f.split('_')
        if len(split) <  4:
            continue
        label = int(split[0].replace("a",""))

        if label >= 23:
            label = label - 2
        else:
            label = label - 1
        #label = label - 1
        if(split[2] == "t1"):
            val_set.append([os.path.join(dirName, f),label])
        else:
            train_set.append([os.path.join(dirName, f),label])

    return train_set,val_set

def write_csv(file_list, output_dir):
    with open(output_dir, 'a') as outcsv:
        #configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

        for item in file_list:
            #Write item to outcsv
            writer.writerow([item[0], item[1]])

#data_path = args.data_dir
data_path =os.path.join(os.getcwd(),"UTD/RGB")
out_path = os.path.join(data_path,"short")
if not os.path.exists(out_path):
    os.mkdir(out_path)

# download my own running/jogging data under /UTD/run_avi
download_avi_files()
own_data = os.path.join(os.getcwd(),"UTD/jogging_avi")

# Generate a shorter video at /UTD/RGB/short
generate_subclip(data_path,out_path)

# Generate list of FilePath
train_set, val_set = load_video_generate_csv(out_path)
my_train_set, my_val_set = load_video_generate_csv(own_data)
train_set = train_set + my_train_set
val_set = val_set + my_val_set

train_csv = os.path.join(data_path,"train.csv")
val_csv = os.path.join(data_path,"val.csv")
test_csv = os.path.join(data_path,"test.csv")

write_csv(train_set,train_csv)
write_csv(val_set,val_csv)
write_csv(val_set,test_csv)



