from pathlib import Path
import json
from tokenize import group
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
# from yolov7.utils.general import increment_path
import re
import glob

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
    

def save_result(opt, all_labels):

    save_dir = Path(Path(os.path.join(opt.output, opt.name), exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir

    # Save json
    # with open(save_dir / "labels.json", 'w', encoding='utf-8') as f:
    #     json.dump(all_labels, f)
    
    for img_name in all_labels:
        version = "1"
        flags = dict()
        shapes = []
        for s in all_labels[img_name]:
            group_id = s["category_id"]
            if group_id == 3:
                group_id = 4
            elif group_id == 2:
                group_id = 3
            elif group_id == 1:
                group_id = 2
            
            shapes.append(
                dict(
                    label=s["text"],
                    points=[[s["x_min"], s["y_min"]], [s["x_max"], s["y_min"]], 
                        [s["x_max"], s["y_max"]], [s["x_min"], s["y_max"]]],
                    group_id=group_id,
                    shape_type="polygon",
                    flags={},
                    group_name=s["category"],
                    det_conf=s["det_conf"],
                    poly = s['poly'],
                )
            )
        saved_dict = dict(
            version=version,
            flags=flags,
            shapes=shapes
        )
        # Save json
        with open(save_dir / 'labels' / (Path(img_name).stem+".json"),
         'w', encoding='utf-8') as f:
            json.dump(saved_dict, f)

    for img_name in all_labels:
        source_path = Path(opt.source)
        if source_path.is_file():
            img_path = source_path
        else:
            img_path = source_path / img_name
        image = cv2.imread(str(img_path))

        for label in all_labels[img_name]:
            category = label['category']
            category_id = label['category_id']
            text = label['text']
            det_conf = label['det_conf']
            x_min = label['x_min']
            y_min = label['y_min']
            x_max = label['x_max']
            y_max = label['y_max']

            points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            if   category_id == 0: color = (0, 255, 0)
            elif category_id == 1: color = (0, 0, 255)
            elif category_id == 2: color = (255, 0, 0)
            elif category_id == 3: color = (255, 255, 0)
            elif category_id == 4: color = (0, 255, 255)
            else:                  color = (255, 0, 255)

            image = cv2.polylines(image, [np.array(points).astype(int)], True, color, 2)
            
            if opt.font != '':
                image_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(image_pil)
                draw.text((points[0][0] - 20, points[0][1] - 20), text, 
                        font=ImageFont.truetype(str(Path(opt.font)), 32), fill=color)
                image = np.array(image_pil)
        cv2.imwrite(str(save_dir/img_name), image)
