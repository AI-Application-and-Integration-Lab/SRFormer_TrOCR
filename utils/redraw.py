import cv2
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

img_folder_path = './datasets/d503_SRFormer_format/test'
label_path = "./TrOCR/output/d503/labels"

folder = os.listdir(label_path)

print(len(folder))

redraw_path = "./final"
os.makedirs(redraw_path, exist_ok=True)
font = "./utils/NotoSansTC-VariableFont_wght.ttf"

for name in folder:
   
   f = open(os.path.join(label_path, name))
   data = json.load(f)

   img = cv2.imread(os.path.join(img_folder_path, name[:-5]+".png"))
   
   for i in range(len(data["shapes"])):
   
      bbox = data["shapes"][i]['points'][0]
      bbox2 = data["shapes"][i]['points'][2]

      poly = np.array(data['shapes'][i]['poly']).reshape(-1,2).astype(np.int32)
   
      img = cv2.polylines(img, pts=[poly], isClosed=True, color=(0,0,255), thickness=2)

      text = data["shapes"][i]['label']
      image_pil = Image.fromarray(img)
      draw = ImageDraw.Draw(image_pil)
      draw.text((int(bbox[0]) - 20, int(bbox[1]) - 20), text, 
            font=ImageFont.truetype(str(Path(font)), 16), fill=(0,0,255))
      
      img = np.array(image_pil)

   cv2.imwrite(os.path.join(redraw_path, name[:-5]+".png"), img)


