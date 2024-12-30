import math
import subprocess
import cv2
import os
import torch
import random
import numpy as np
import time

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to extract I-frames and their frame numbers using ffprobe
def extract_iframes(video_path):
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return [], []

    # Command to get frame types(I,B,P) from video
    command = [
        'ffprobe', '-select_streams', 'v:0', '-show_frames', '-show_entries',
        'frame=pict_type', '-of', 'csv', video_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return [], []

    iframe_numbers = []
    frame_count = 0  
    
    for line in lines:
        if 'I' in line:
            iframe_numbers.append(frame_count)
        frame_count += 1

    return iframe_numbers

# Function to get total frame count in a video using ffprobe
def get_total_frames(video_path):
    command = [
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting total frame count: {e.stderr}")
        return 0

# Function to run object detection on a single frame and return object names and confidence scores
def run_object_detection_on_frame(frame_path):
    frame = cv2.imread(frame_path)
    # Run YOLO model on frame
    results = model(frame)
    detected = results.pandas().xyxy[0] 
    return list(zip(detected['name'], detected['confidence']))

# Function to perform Index Construction by sampling I-frames
def index_construction(input_video, iframe_numbers, N, alpha):
    sample_size = int(alpha * N)
    if sample_size >= len(iframe_numbers):
        sampled_numbers = iframe_numbers
    else:
        indices = random.sample(range(len(iframe_numbers)), sample_size)
        sampled_numbers = [iframe_numbers[i] for i in indices]
    
    sampled_numbers.sort()
    sampled_iframes = []
    for n in sampled_numbers:
        frame_filename = f"I_frame_{n}.png"
        # Use ffmpeg to extract the I-frame
        command = [
            'ffmpeg', '-i', input_video, '-vf', f"select='eq(n\\,{n})'",
            '-frames:v', '1', frame_filename
        ]
        try:
            subprocess.run(command, check=True)
            sampled_iframes.append(frame_filename)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frame {n}: {e.stderr}")
            continue
    
    print(f"Sampled {len(sampled_numbers)} I-frames based on sampling ratio {alpha} and budget {N}.")
    
    return sampled_iframes, sampled_numbers

# Function to sample additional samples using MAB sampling
def query_execution_phase(budget, iframe_numbers, query_type, object_of_interest, threshold, video_path):
    total_frames = get_total_frames(video_path)
    if total_frames == 0:
        print("Error: Unable to determine total frames in the video.")
        return {}

    num_rounds = budget - len(iframe_numbers)
    print("budget:", budget, " num_rounds:", num_rounds)
    num_arms = len(iframe_numbers) + 1
    count = np.zeros(num_arms)
    arm_rewards = [[] for _ in range(num_arms)]
    Q = np.zeros(num_arms)

    def cal_reward(frame_filename, query, req_object, threshold):
        reward = 0
        detected_objs = run_object_detection_on_frame(frame_filename)
        if query == "retrieval":
            for obj, conf in detected_objs:
                if obj == req_object and conf >= threshold:
                    reward = 1
                    break
        elif query == "aggregate":
            for obj, conf in detected_objs:
                if obj == req_object:
                    reward += 1
        return reward

    def UCB(iters):
        ucb = np.zeros(num_arms)
        
        if iters < num_arms:
            return iters  # Explore all arms in the initial iterations
        else:
            for arm in range(num_arms):
                if count[arm] > 0:
                    upper_bound = math.sqrt((2 * math.log(sum(count))) / count[arm])
                else:
                    upper_bound = float('inf')  # Explore arms with no pulls
                ucb[arm] = Q[arm] + upper_bound
            return np.argmax(ucb)

    additional_sample_nums = []
    additional_sample_paths = []

    for i in range(num_rounds):
        print(f"Round: {i}\n")
        arm = UCB(i)

        start_ind = 0 if arm == 0 else iframe_numbers[arm - 1]
        end_ind = iframe_numbers[arm] if arm < len(iframe_numbers) else total_frames - 1

        sample_frame_number = random.randint(start_ind, end_ind)
        if sample_frame_number >= total_frames:
            print(f"Invalid frame number {sample_frame_number}. Skipping.")
            continue

        frame_filename = f"I_frame_{sample_frame_number}.png"
        command = [
            'ffmpeg', '-i', video_path, '-vf', f"select='eq(n\\,{sample_frame_number})'",
            '-frames:v', '1', frame_filename
        ]
        try:
            subprocess.run(command, check=True)
            additional_sample_paths.append(frame_filename)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frame {sample_frame_number}: {e.stderr}")
            continue

        reward = cal_reward(frame_filename, query_type, object_of_interest, threshold)
        
        count[arm] += 1
        arm_rewards[arm].append(reward)
        Q[arm] = np.mean(arm_rewards[arm]) if arm_rewards[arm] else 0
    
    additional_index = store_inference_results(additional_sample_paths, additional_sample_nums)
    return additional_index

# Function to store inference results in a key-value index
def store_inference_results(i_frame_images, iframe_numbers):
    inference_index = {}
    for frame_path, frame_number in zip(i_frame_images, iframe_numbers):
        detected_objects = run_object_detection_on_frame(frame_path)
        inference_index[frame_number] = detected_objects
    return inference_index

# Aggregate query: Average count of an object in I-frames
def execute_aggregate_query(inference_index, object_of_interest):
    total_object_count = 0
    for detected_objects in inference_index.values():
        total_object_count += sum(1 for obj, _ in detected_objects if obj == object_of_interest)
    return total_object_count / len(inference_index) if inference_index else 0

# Retrieval query: Get I-frames containing a specific object, along with confidence scores
def execute_retrieval_query(inference_index, object_of_interest, confidence_threshold=0.5):
    relevant_frames = []
    for frame_number, detected_objects in inference_index.items():
        for obj, conf in detected_objects:
            if obj == object_of_interest and conf >= confidence_threshold:
                relevant_frames.append((frame_number, conf)) 
                break
    return relevant_frames

# Main function to process the video input
def process_video(video_path, N, alpha):
    print("Extracting I-frames...")
    iframe_numbers = extract_iframes(video_path)
    
    if not iframe_numbers:
        print("No I-frames found or failed to extract I-frames.")
        return

    print(f"Extracted {len(iframe_numbers)} I-frames.")

    sampled_iframes, sampled_numbers = index_construction(video_path, iframe_numbers, N, alpha)

    start_time = time.time()
    
    print("Running object detection on sampled I-frames and storing results...")
    inference_index = store_inference_results(sampled_iframes, sampled_numbers)
    for key, value in inference_index.items():
        print(key, ": ", value, "\n")

    query_type = "aggregate"
    object_of_interest = "car"
    confidence_threshold = 0
    if query_type == 'retrieval':
        confidence_threshold = float(input("Enter confidence threshold (e.g., 0.5): "))

    additional_index = query_execution_phase(N, sampled_numbers, query_type, object_of_interest, confidence_threshold, video_path)
    
    total_samples = {**inference_index, **additional_index}

    if query_type == 'aggregate':
        avg_count = execute_aggregate_query(total_samples, object_of_interest)
        print(f"Average {object_of_interest} count per frame: {avg_count}")

    elif query_type == 'retrieval':
        relevant_frames = execute_retrieval_query(total_samples, object_of_interest, confidence_threshold)
        print(f"I-frames containing {object_of_interest} with confidence >= {confidence_threshold}: {relevant_frames}")

    else:
        print("Invalid query type. Please enter either 'aggregate' or 'retrieval'.")

    end_time = time.time()

    # Clean up I-frame images
    for frame in sampled_iframes:
        frame_path = os.path.join(os.getcwd(), frame)
        if os.path.exists(frame_path): 
            os.remove(frame_path)
    print("Cleaned up I-frame images.")

    return (end_time-start_time),avg_count

videos=["/kaggle/input/24-min-video/24-min-video.mp4"]

N=[40]
alpha = 0.5
t=[]
a=[]

if __name__ == "__main__":
    for i in range(0,len(N)):
        t1,a1=process_video(videos[i], N[i], alpha)
        t.append(t1)
        a.append(a1)
    print("Time taken and the outputs for every video")
    for i in range(0,len(N)):
        print(f"For video {videos[i]}:\nTime:{t[i]}\nOutput:{a[i]}")
