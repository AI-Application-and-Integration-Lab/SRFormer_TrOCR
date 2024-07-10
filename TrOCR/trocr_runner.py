from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from pathlib import Path
from tqdm import tqdm
import torch

class TrOCRRunner:
    def __init__(self, opt):
        self.opt = opt
        self.processor = TrOCRProcessor.from_pretrained(self.opt.recog_model)
        self.device = torch.device('cuda:0')
        self.model = VisionEncoderDecoderModel.from_pretrained(self.opt.recog_model).to(self.device)
    
    def run(self, all_bboxes):
        print("Run Recognition...")
        all_labels = dict()
        source = self.opt.source

        for img_name in tqdm(all_bboxes):
            source_path = Path(source)
            if source_path.is_file():
                img_path = source_path
            else:
                img_path = source_path / img_name
            # Read image
            img = Image.open(img_path)
            w, h = img.size

            labels = []
            for bbox in all_bboxes[img_name]:
                category, category_id, x_min, y_min, x_max, y_max, polygons \
                    = bbox[0], int(bbox[1]), float(bbox[2]), \
                        float(bbox[3]), float(bbox[4]), float(bbox[5]), bbox[6]
                if len(bbox) == 8:
                    conf = float(bbox[7])
                else:
                    conf = None
                
                # x_center *= w
                # y_center *= h
                # width *= w
                # height *= h

                # x_min = max(0, int(round(x_center - (width / 2))))
                # x_max = min(w, int(round(x_center + (width / 2))))
                # y_min = max(0, int(round(y_center - (height / 2))))
                # y_max = min(h, int(round(y_center + (height / 2))))

                try:
                    # Crop image
                    crop_img = img.crop((x_min, y_min, x_max, y_max))
                    # Processor
                    pixel_values = self.processor(images=crop_img, return_tensors="pt").pixel_values.to(self.device)
                    generated_ids = self.model.generate(pixel_values)
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    labels.append(dict(category = category,
                                        category_id = category_id,
                                        x_min = x_min,
                                        y_min = y_min,
                                        x_max = x_max,
                                        y_max = y_max,
                                        det_conf = conf,
                                        text = generated_text, poly=polygons))
                except:
                    pass
            
            all_labels[img_name] = labels
            
        return all_labels