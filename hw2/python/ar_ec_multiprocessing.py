import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac
from planarH import compositeH
import subprocess as sp
import multiprocessing as mp
#from helper import plotMatches
opts = get_opts()

num_processes = mp.cpu_count()
print("Number of CPU: " + str(num_processes))
frame_count = 511
frame_jump_unit =  frame_count// num_processes

path_book = "../data/book.mov"
path_ars = "../data/ar_source.mov"

capbook = cv2.VideoCapture(path_book)
capars = cv2.VideoCapture(path_ars)

cv_cover = cv2.imread('../data/cv_cover.jpg')
#cv_cover_gray = cv2.cvtColor(cv_cover,cv2.COLOR_BGR2GRAY)


def process_video_multiprocessing(group_number):
    # Read video file
    #cap = cv.VideoCapture(file_name)

    #cap.set(cv.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)
    capbook.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)
    capars.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)
    
    # get height, width and frame count of the video
    #width, height = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    
    no_of_frames = int(capbook.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capbook.get(cv2.CAP_PROP_FPS))
    proc_frames = 0

    # Define the codec and create VideoWriter object
    #fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    #out = cv.VideoWriter()
    #output_file_name = "output_multi.mp4"
    #out.open("output_{}.mp4".format(group_number), fourcc, fps, (width, height), True)
    try:
        while proc_frames < 1:
            retbook, framebook = capbook.read()
            retars, framears = capars.read()
            if not retbook or not retars:
                break

            centerx = framears.shape[1]//2
            x = centerx - cv_cover.shape[1]//2
            framears = framears[45:315, int(x):int(x+cv_cover.shape[1]),:]
            framears = cv2.resize(framears, (cv_cover.shape[1], cv_cover.shape[0]))           
            
            matches, locs1, locs2 = matchPics(framebook, cv_cover, opts)
            
            locs1 = locs1[matches[:,0]]
            locs1 = locs1[:,[1,0]]
            locs2 = locs2[matches[:,1]]
            locs2 = locs2[:,[1,0]]
            bestH2to1, inliers = computeH_ransac(locs1, locs2, opts) #tell us how to go from locs2 to locs1
            
            comp_img = compositeH(bestH2to1, framears, framebook)
            dst = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
            
            # displaying the frame with fps
            cv2.imshow('frame', dst[:,:,::-1])
            proc_frames += 1
            print(proc_frames())
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except:
        # Release resources
        capbook.release()
        capars.release()

    # Release resources
    capbook.release()
    capars.release()
    
##################################################


#def combine_output_files(num_processes):
#    # Create a list of output files and store the file names in a txt file
#    list_of_output_files = ["output_{}.mp4".format(i) for i in range(num_processes)]
#    with open("list_of_output_files.txt", "w") as f:
#        for t in list_of_output_files:
#            f.write("file {} \n".format(t))
#
#    # use ffmpeg to combine the video output files
#    ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + output_file_name
#    sp.Popen(ffmpeg_cmd, shell=True).wait()
#
#    # Remove the temperory output files
#    for f in list_of_output_files:
#        remove(f)
#    remove("list_of_output_files.txt")
###################################################
def multi_process():
    print("Video processing using {} processes...".format(num_processes))
    start_time = time.time()

    # Paralle the execution of a function across multiple input values
    p = mp.Pool(num_processes)
    p.map(process_video_multiprocessing, range(num_processes))

    #combine_output_files(num_processes)

    end_time = time.time()

    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))
    print("FPS : {}".format(frame_count/total_processing_time))

file_name = "input.mp4"
output_file_name = "output.mp4"
#width, height, frame_count = get_video_frame_details(file_name)
#print("Video frame count = {}".format(frame_count))
#print("Width = {}, Height = {}".format(width, height))
multi_process()
cv2.destroyAllWindows()