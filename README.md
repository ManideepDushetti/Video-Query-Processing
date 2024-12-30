# Seiden: Query Processing in Video Database Systems

Seiden is an advanced algorithm designed to process queries efficiently in large-scale video databases. It uses innovative sampling strategies and object detection models to deliver fast and reliable query results. This implementation is inspired by the [Georgia Tech Database project](https://github.com/georgia-tech-db/seiden_submission).

## Features
- Extracts I-frames for efficient indexing.
- Uses YOLOv5 for real-time object detection.
- Supports retrieval and aggregate queries.
- Implements multi-armed bandit (MAB) sampling for optimized query execution.

## Prerequisites
To run this code, ensure you have the following installed:
- Python 3.8 or later
- [PyTorch](https://pytorch.org/) (compatible with your system)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- OpenCV
- FFmpeg
- Kaggle account for running the code in a Kaggle environment (optional)

Install dependencies:
```bash
pip install torch torchvision numpy opencv-python-headless pandas
```

Ensure FFmpeg is installed and accessible in your system's PATH.

## Getting Started

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/seiden.git
cd seiden
```

### Step 2: Prepare Your Video Dataset
Place the video files in a folder and specify their paths in the `videos` list within the script.

### Step 3: Run the Code
Run the script using Python:
```bash
python seiden.py
```

## Parameters
- **`N`**: Total sampling budget for I-frames.
- **`alpha`**: Ratio of budget allocated for Index Construction phase.
- **`query_type`**: Type of query (`'aggregate'` or `'retrieval'`).
- **`object_of_interest`**: Object to query (e.g., `'car'`, `'bus'`).
- **`confidence_threshold`**: Minimum confidence for retrieval queries.

## Outputs
- Aggregate Query: Returns the average count of the specified object across frames.
- Retrieval Query: Lists I-frames containing the specified object along with confidence scores.

## Code Details
### Index Construction
- Extracts I-frames using FFmpeg and samples them based on the ratio `alpha`.

### Query Execution
- Utilizes MAB sampling to identify additional frames of interest.
- Applies YOLOv5 for object detection.

### Key Functions
- `extract_iframes(video_path)`
- `index_construction(input_video, iframe_numbers, N, alpha)`
- `query_execution_phase(budget, iframe_numbers, query_type, object_of_interest, threshold, video_path)`
- `store_inference_results(i_frame_images, iframe_numbers)`
- `execute_aggregate_query(inference_index, object_of_interest)`
- `execute_retrieval_query(inference_index, object_of_interest, confidence_threshold)`

## Example Usage
```python
videos = ["/path/to/video.mp4"]
N = [40]
alpha = 0.5

process_video(videos[0], N[0], alpha)
```

## References
This project is inspired by the original work at [Georgia Tech DB](https://github.com/georgia-tech-db/seiden_submission).

