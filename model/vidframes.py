import cv2 #opencv-python==4.6.0
import os
import math
import shutil

# def remove_output_folder():
#     if os.path.exists('image_dir'):
#         shutil.rmtree('image_dir')
        
# def create_output_folder():
#     if not os.path.exists('image_dir'):
#         os.mkdir('image_dir')

def capture_video(vidname, fps):
    cap = cv2.VideoCapture(vidname)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to extract (maximum 8 frames)
    num_frames_to_extract = 8

    # Calculate the bin size for equal binning
    bin_size = total_frames // num_frames_to_extract
    remainder = total_frames % num_frames_to_extract

    frames_list = []

    for count in range(num_frames_to_extract):
        # Calculate the frame index for the current bin
        frameId = bin_size * count + min(count, remainder)

        # Jump to the desired frame
        cap.set(1, frameId)

        # Read the frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Append the frame to the list
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame)

    cap.release()

    return frames_list


def extract(file, fps=1):
#     remove_output_folder()
#     create_output_folder()
    frames = capture_video(file, fps) #video name and number of seconds for 1 frame
    return frames