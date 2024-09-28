import subprocess
import cv2
import os
import torch
import glob
import random

# Load YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def extract_iframes(input_video, N):
    # Check if the video exists
    if not os.path.exists(input_video):
        print(f"Error: File '{input_video}' not found.")
        return [], []

    # Get video duration
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                             'format=duration', '-of',
                             'default=noprint_wrappers=1:nokey=1', input_video],
                             capture_output=True, text=True)
    duration = float(result.stdout.strip())

    # Calculate random times to sample I-frames
    times = sorted(random.sample(range(int(duration)), N))

    iframe_images = []
    iframe_numbers = []  # This will now hold the timestamps for the I-frames
    for t in times:
        frame_filename = f"I_frame_at_{t}.png"
        command = [
            'ffmpeg', '-ss', str(t), '-i', input_video, '-frames:v', '1', '-q:v', '2',
            '-vf', "select='eq(pict_type\\,PICT_TYPE_I)'", '-vsync', 'vfr', frame_filename
        ]
        subprocess.run(command, check=True)
        if os.path.exists(frame_filename):
            iframe_images.append(frame_filename)
            iframe_numbers.append(t)  # Store the time as the frame number

    return iframe_images, iframe_numbers

# Function to run object detection on a single frame and return object names and confidence scores
def run_object_detection_on_frame(frame_path):
    frame = cv2.imread(frame_path)
    # Run YOLOv5 inference on the frame
    results = model(frame)
    detected = results.pandas().xyxy[0]  # Extract the detection results
    # Extract object names and confidence scores as a list of tuples
    return list(zip(detected['name'], detected['confidence']))

# Function to perform Index Construction by sampling I-frames
def index_construction(i_frame_images, iframe_numbers, N, alpha):
    # Determine the number of frames to sample based on alpha and N
    sample_size = int(alpha * N)

    # If the sample size is greater than the number of I-frames, select all I-frames
    if sample_size >= len(i_frame_images):
        sampled_iframes = i_frame_images
        sampled_numbers = iframe_numbers
    else:
        # Randomly sample the frames
        indices = random.sample(range(len(i_frame_images)), sample_size)
        sampled_iframes = [i_frame_images[i] for i in indices]
        sampled_numbers = [iframe_numbers[i] for i in indices]

    print(f"Sampled {len(sampled_iframes)} I-frames based on sampling ratio {alpha} and budget {N}.")

    return sampled_iframes, sampled_numbers

# Function to store inference results in a key-value index
# Key = Frame Number, Value = List of (object, confidence) tuples
def store_inference_results(i_frame_images, iframe_numbers):
    inference_index = {}
    for frame_path, frame_number in zip(i_frame_images, iframe_numbers):
        detected_objects = run_object_detection_on_frame(frame_path)
        # Store the frame number and its detected objects with confidence scores in the index
        inference_index[frame_number] = detected_objects
    return inference_index

# Aggregate query: Average count of an object in I-frames
def execute_aggregate_query(inference_index, object_of_interest):
    total_object_count = 0
    for frame_number, detected_objects in inference_index.items():
        # Count how many times the object of interest appears in each frame
        total_object_count += sum(1 for obj, conf in detected_objects if obj == object_of_interest)

    return total_object_count / len(inference_index)  # Average object count per frame

# Retrieval query: Get I-frames containing a specific object, along with confidence scores
def execute_retrieval_query(inference_index, object_of_interest, confidence_threshold=0.5):
    relevant_frames = []
    for frame_number, detected_objects in inference_index.items():
        # Check if the object is in the frame with a confidence greater than or equal to the threshold
        for obj, conf in detected_objects:
            if obj == object_of_interest and conf >= confidence_threshold:
                relevant_frames.append((frame_number, conf))  # Store frame number and confidence score
                break  # Move to the next frame once the object is found

    return relevant_frames  # Return list of (frame number, confidence) tuples containing the object

# Main function to process the video input
def process_video(video_path, N, alpha):
    # Step 1: Extract I-frames using ffprobe
    print("Extracting I-frames...")
    i_frame_images, iframe_numbers = extract_iframes(video_path,N)

    if not i_frame_images:
        print("No I-frames found or failed to extract I-frames.")
        return

    print(f"Extracted {len(i_frame_images)} I-frames.")

    # Step 2: Perform Index Construction (sampling I-frames based on N and alpha)
    sampled_iframes, sampled_numbers = index_construction(i_frame_images, iframe_numbers, N, alpha)

    # Step 3: Run inference on each sampled frame and store results in a key-value index
    print("Running object detection on sampled I-frames and storing results...")
    inference_index = store_inference_results(sampled_iframes, sampled_numbers)
    for key, value in inference_index.items():
        print(key, ": ", value, "\n")
    # Prompt the user to select query type and object of interest
    query_type = input("Enter query type (aggregate/retrieval): ").lower().strip()
    object_of_interest = input("Enter the object of interest (e.g., car, truck, person): ").strip().lower()

    if query_type == 'retrieval':
        confidence_threshold = float(input("Enter confidence threshold (e.g., 0.5): "))

    # Step 4: Run the selected query
    if query_type == 'aggregate':
        # Execute aggregate query (average count of objects)
        avg_count = execute_aggregate_query(inference_index, object_of_interest)
        print(f"Average {object_of_interest} count per frame: {avg_count}")
    elif query_type == 'retrieval':
        # Execute retrieval query (get frames with the object, filtered by confidence)
        relevant_frames = execute_retrieval_query(inference_index, object_of_interest, confidence_threshold)
        print(f"I-frames containing {object_of_interest} with confidence >= {confidence_threshold}: {relevant_frames}")
    else:
        print("Invalid query type. Please enter either 'aggregate' or 'retrieval'.")

    # Clean up: Delete extracted I-frame images after processing
    for frame in sampled_iframes:
        os.remove(frame)
    print("Cleaned up I-frame images.")

# Run the script
if __name__ == "__main__":
    video_path = "C:\\Users\\manid\\OneDrive\\Desktop"
    N = 50
    alpha = 0.5
    process_video(video_path, N, alpha)

