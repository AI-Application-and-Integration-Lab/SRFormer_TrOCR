import cv2
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='./datasets/d503_SRFormer_format/test', help='source') 
parser.add_argument('--label', type=str, default="./TrOCR/output/d503/labels", help='source') 
parser.add_argument('--final', type=str, default="./d503_final", help='source') 
opt = parser.parse_args()

img_folder_path = opt.image   # './datasets/d503_SRFormer_format/test'
label_path = opt.label        # "./TrOCR/output/d503/labels"


folder = os.listdir(label_path)

print(len(folder))

redraw_path = opt.final
os.makedirs(redraw_path, exist_ok=True)
# font = "./utils/NotoSansTC-VariableFont_wght.ttf"
font = ImageFont.truetype("./utils/NotoSansTC-VariableFont_wght.ttf", 16)
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
      position = (int(bbox[0]) - 20, int(bbox[1]) - 20)
      
      draw = ImageDraw.Draw(image_pil)
      bbox = draw.textbbox(position, text, font)
      
      draw.rectangle(bbox, fill=(0,0,255,128))
      draw.text(position, text, font=font, fill=(0,0,0))
      
      img = np.array(image_pil)

   cv2.imwrite(os.path.join(redraw_path, name[:-5]+".png"), img)


