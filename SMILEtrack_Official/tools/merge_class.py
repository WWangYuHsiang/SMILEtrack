# 导入需要用到的库
import os


def calculate_iou(bb1, bb2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes.
    # bb1 and bb2 should be [left, top, width, height].
    [int(bb1) for bb1 in a]
    [int(bb2) for bb2 in a]
    
    # Get the coordinates of intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    union_area = bb1_area + bb2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

# 输入文件路径和输出文件路径
input_file_path1 = "trk_Fc7cf07iou65.txt"
input_file_path2 = "trk_Fc3cf07iou65.txt"
output_file_path = "merge_cl3_cl7.txt"

# 设置边界框大小和缩小像素值
# 在这里我设置的边界框大小是(1280, 720)，缩小像素值为1，你需要根据你的实际情况进行修改
boundary_box = (1920, 1080)
shrink_pixel = 1

# 读取输入文件并过滤掉超出边界的物体
with open(input_file_path1, "r") as f_input1:
    for line1 in f_input1:
        object_info1 = line1.strip().split(",")
        video_id_1, frame_id_1 = map(float, object_info1[0:2])
        with open(output_file_path, "w") as f_output:
            with open(input_file_path2, "r") as f_input2:
                for line2 in f_input2:
                    object_info2 = line2.strip().split(",")
                    video_id_2, frame_id_2 = map(float, object_info2[0:2])
                    iou = calculate_iou(object_info1[2:6],object_info2[2:6])
                    if (video_id_1 ==video_id_2 and frame_id_1 == frame_id_2 and iou >95)!= True:
                        f_output.write(",".join(map(str, object_info2[:])) + "\n")
        f_output.write(",".join(map(str, object_info1[:])) + "\n")


