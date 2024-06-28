import os
import json
import cv2
from tqdm import tqdm
import numpy as np

img_path = './datasets/d503/train/images'
img_folder = os.listdir(img_path)

annotation_path = "./datasets/d503/train/labels"
annotation_folder = os.listdir(annotation_path)

tgt_folder_name = 'd503_SRFormer_format'

img_tgt_path = f"./datasets/{tgt_folder_name}/train"
label_tgt_path = f"./datasets/{tgt_folder_name}"

os.makedirs(img_tgt_path, exist_ok=True)

# 將我們的 D503 資料集轉換成 COCO 的標註形式,以供 SRFormer 訓練時使用。
def cal_bbox(points):
    
    sorted_points = sorted(points, key=lambda x: x[0])   
    top_point = sorted(sorted_points[:2], key=lambda x: x[1])
    bottom_point = sorted(sorted_points[2:], key=lambda x: x[1])
    left_top_point = top_point[0]
    right_top_point = top_point[1]
    left_bottom_point = bottom_point[0]
    right_bottom_point = bottom_point[1]
    [h, w] = [right_bottom_point[i] - left_top_point[i] for i in range(2)]

    return [round(float(left_top_point[0]),2), round(float(left_top_point[1]),2), float(h), float(w)], int(w*h), [left_top_point, right_top_point, right_bottom_point, left_bottom_point]

# 去取上面8個點，下面8個點。
def cal_polys(points):

    sorted_points = sorted(points, key=lambda x: x[0])   
    top_point = sorted(sorted_points[:2], key=lambda x: x[1])
    bottom_point = sorted(sorted_points[2:], key=lambda x: x[1])
    
    left_top_point = top_point[0]
    left_bottom_point = top_point[1]
    right_top_point = bottom_point[0]
    right_bottom_point = bottom_point[1]
        
    top_x_step = float((right_top_point[0]-left_top_point[0])/7)
    top_y_step = float((right_top_point[1]-left_top_point[1])/7)

    bottom_x_step = float((left_bottom_point[0]-right_bottom_point[0])/7)
    bottom_y_step = float((left_bottom_point[1]-right_bottom_point[1])/7)

    poly = []
    for i in range(8):
        x = left_top_point[0] + top_x_step*i
        y = left_top_point[1] + top_y_step*i
        poly.append(round(x, 2))
        poly.append(round(y, 2))
    
    for i in range(8):
        
        x = right_bottom_point[0] + bottom_x_step*i
        y = right_bottom_point[1] + bottom_y_step*i        
        
        poly.append(round(float(x),2))
        poly.append(round(float(y),2))

    return poly

images = []
annotations = []
label_id = 0

for id, image_name in enumerate(tqdm(img_folder)):
    image = cv2.imread(os.path.join(img_path, image_name))
    cv2.imwrite(os.path.join(img_tgt_path, image_name), image)
    
    height, width, _ = image.shape

    f = open(os.path.join(annotation_path, image_name[:-4]+'.json'))
    image_dict = dict()
    image_dict["file_name"] = image_name
    image_dict["height"] = height
    image_dict["width"] = width
    image_dict["id"] = id
    image_dict["license"] = 0
    images.append(image_dict)

    current_json = json.load(f)

    # if group_id ==2 > eng str, group_id == 0 >> chinese str

    for idx, label in enumerate(current_json["shapes"]):
        if label['group_id'] == 0 or label['group_id']==2:
            
            label_dict = dict()
            label_dict["iscrowd"] = 0
            label_dict["category_id"] = 1 # 固定使用 1
            label_dict["bbox"], label_dict['area'], points = cal_bbox(label['points'])        
            label_dict["polys"] = cal_polys(label["points"])
            label_dict["image_id"] = id
            label_dict["id"] = label_id; label_id+=1
            annotations.append(label_dict)
        
all_dict = dict()
all_dict['images'] = images
all_dict['categories'] = [dict(id=1, name='text')]
all_dict['annotations'] = annotations

with open( os.path.join(label_tgt_path, "train_poly_pos.json"), "w") as outfile:
    json.dump(all_dict, outfile)