from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
import os
import os.path
#path_to_vid="/scratch/user/yiminchou1994/ActivityNet/Crawler/Kinetics/video/small_val/jogging/0S8erRQSUZs_000093_000103.mp4"
import csv
import numpy as np
from moviepy.editor import VideoFileClip
def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

def generate_subclip(cfg):
    file_path = cfg.TEST.DEMO_PATH
    file_name = file_path.split("/")[-1].split(".")[0]

    OUTPUT_DIR = os.path.abspath(os.path.join(file_path, os.pardir))
    OUTPUT_DIR = os.path.join(OUTPUT_DIR ,"demo")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print("Directory " , OUTPUT_DIR ,  " Created ")
    else:    
        print("Directory " , OUTPUT_DIR ,  " already exists")
    duration = (get_length(file_path))
    #clip = VideoFileClip(file_path)
    #duration2 = clip.duration
    #print(duration2)
    file_name_list = []
    end_time_list=[]
    for i in np.arange(0.25,duration+0.24,0.25):
        print(duration,i)
        
        start_time = max(0,i-1)
        end_time = min(i,duration)
        end_time_list.append(end_time)
        target_name = os.path.join(OUTPUT_DIR, file_name + "_" + str(start_time) + ".mp4")
        ffmpeg_extract_subclip(file_path, start_time, end_time, targetname=target_name)
        file_name_list.append([target_name,1])

    cfg.DATA.PATH_TO_DATA_DIR = OUTPUT_DIR
    csv_out = os.path.join(OUTPUT_DIR,"test.csv")
    npy_out = os.path.join(OUTPUT_DIR,"end_time.npy")
    if os.path.exists(csv_out):
        os.remove(csv_out)
    if os.path.exists(npy_out):
        os.remove(npy_out)
    np.save(npy_out,np.array(end_time_list))
    
    with open(csv_out, 'a') as outcsv:
        #configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

        for item in file_name_list:
            #Write item to outcsv
            writer.writerow([item[0], item[1]])

    return


