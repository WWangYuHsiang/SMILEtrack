import csv
from collections import defaultdict
import numpy as np

def read_txt_file(filename):
    data = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            video_id, frame_id, object_id, left, top, width, height, class_, confidence = row
            data[object_id].append((int(video_id), int(frame_id), float(left), float(top), float(width), float(height), class_, float(confidence)))
    return data

def interpolate(data):
    interpolated_data = defaultdict(list)
    for object_id, object_data in data.items():
        object_data.sort(key=lambda x: x[1])  # Sort by frame_id
        video_ids, frame_ids, lefts, tops, widths, heights, classes, confidences = zip(*object_data)

        for i in range(1, len(frame_ids)):
            frame_diff = frame_ids[i] - frame_ids[i-1]
            if frame_diff > 1:
                for j in range(1, frame_diff):
                    t = j / frame_diff
                    interpolated_frame_id = frame_ids[i-1] + j
                    interpolated_left = lefts[i-1] + t * (lefts[i] - lefts[i-1])
                    interpolated_top = tops[i-1] + t * (tops[i] - tops[i-1])
                    interpolated_width = widths[i-1] + t * (widths[i] - widths[i-1])
                    interpolated_height = heights[i-1] + t * (heights[i] - heights[i-1])
                    interpolated_confidence = confidences[i-1] + t * (confidences[i] - confidences[i-1])

                    interpolated_data[object_id].append((video_ids[i], interpolated_frame_id, interpolated_left, interpolated_top, interpolated_width, interpolated_height, classes[i], interpolated_confidence))

        # Add original data to the interpolated data
        interpolated_data[object_id].extend(object_data)
        interpolated_data[object_id].sort(key=lambda x: x[1])  # Sort by frame_id

    return interpolated_data

def save_to_txt(interpolated_data, filename_with_id, filename_without_id):
    with open(filename_with_id, 'w') as f_with_id, open(filename_without_id, 'w') as f_without_id:
        all_data_without_id = []
        for object_id, object_data in interpolated_data.items():
            for frame_data in object_data:
                f_with_id.write(f"{frame_data[0]},{int(frame_data[1])},{object_id},{int(frame_data[2])},{int(frame_data[3])},{int(frame_data[4])},{int(frame_data[5])},{frame_data[6]},{frame_data[7]:.6f}\n")
                all_data_without_id.append(frame_data)
        
        # Sort data without object_id based on video_id and frame_id
        all_data_without_id.sort(key=lambda x: (x[0], x[1]))
        
        for frame_data in all_data_without_id:
            f_without_id.write(f"{frame_data[0]},{int(frame_data[1])},{int(frame_data[2])},{int(frame_data[3])},{int(frame_data[4])},{int(frame_data[5])},{frame_data[6]},{frame_data[7]:.6f}\n")

def main():
    input_file = "AIMOT_result.txt"
    output_file_with_id = "output_with_object_id.txt"
    output_file_without_id = "output_without_object_id.txt"
    data = read_txt_file(input_file)
    interpolated_data = interpolate(data)
    save_to_txt(interpolated_data, output_file_with_id, output_file_without_id)

if __name__ == "__main__":
    main()
