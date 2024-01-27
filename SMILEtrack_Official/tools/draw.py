import cv2
import csv
from collections import defaultdict

def read_output_file(filename):
    data = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            video_id, frame_id, left, top, width, height, class_, confidence = row
            data[int(video_id)].append((int(frame_id), int(left), int(top), int(width), int(height), class_, float(confidence)))
    return data

def draw_bounding_boxes(video_folder, data):
    for video_id, objects in data.items():
        video_path = f"{video_folder}/{video_id:03d}.mp4"
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Video {video_path} not found.")
            continue

        frame_objects = defaultdict(list)
        for obj in objects:
            frame_id, left, top, width, height, class_, confidence = obj
            frame_objects[frame_id].append((left, top, width, height, class_, confidence))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{video_folder}/{video_id:03d}_output.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id in frame_objects:
                for obj in frame_objects[frame_id]:
                    left, top, width, height, class_, confidence = obj
                    pt1 = (left, top)
                    pt2 = (left + width, top + height)
                    color = (0, 255, 0)  # Green color
                    thickness = 2

                    cv2.rectangle(frame, pt1, pt2, color, thickness)

                    text = f"{class_} {confidence:.2f}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_pt1 = (left, top - 5)
                    text_pt2 = (left + text_size[0], top - text_size[1] - 5)
                    cv2.rectangle(frame, text_pt1, text_pt2, color, -1)
                    cv2.putText(frame, text, text_pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            out.write(frame)
            frame_id += 1

        cap.release()
        out.release()

def main():
    output_file = "output_without_object_id.txt"
    video_folder = "videos"
    data = read_output_file(output_file)
    draw_bounding_boxes(video_folder, data)

if __name__ == "__main__":
    main()
