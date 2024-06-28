# TrOCR

## Inference
With [Huggingface-Transformers](https://github.com/huggingface/transformers), the models above could be easily accessed and loaded through the following codes.
```
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("MODEL_NAME")
model = VisionEncoderDecoderModel.from_pretrained("MODEL_NAME")
```
The actual model and its `MODEL_NAME` are listed below.

| Original Model | MODEL_NAME                |
| -------------- | ------------------------- |
| TrOCR-base pre-trained on D501  | ycchen/TrOCR-base-ver021-v1 |
