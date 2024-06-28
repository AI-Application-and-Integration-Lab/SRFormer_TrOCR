import json
import cv2
import os
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='inference/images', help='source')
parser.add_argument('--target', type=str, default='inference/images', help='source')
opt = parser.parse_args()

f = open(opt.source)
data = json.load(f)

transform_data = dict()
count = 0
for d in tqdm(data):
   # img = cv2.imread(d['original_image'])
   # H, W = img.shape[:2]
   key = d['original_image'].split('/')[-1]
   x_min, y_min = d['poly'][0]
   x_max, y_max = d['poly'][8]

   if key in transform_data.keys():
      transform_data[key].append( ['En-Str', '0', x_min, y_min, x_max, y_max, d['poly']])
   else:
      transform_data[key] = [ ['En-Str', '0', x_min, y_min, x_max, y_max, d['poly']]]

with open(opt.target, "w") as outfile:
    json.dump(transform_data, outfile)