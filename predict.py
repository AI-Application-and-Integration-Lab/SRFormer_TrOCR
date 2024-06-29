import argparse, os
# from yolov7.yolov7_runner import Yolov7Runner
from TrOCR.trocr_runner import TrOCRRunner
from utils.visualize import save_result
import json

def main(opt):
    # yolov7 init
    # yolov7 = Yolov7Runner(opt)
    # # Detect text
    # all_bboxes = yolov7.run()
    
    f = open(opt.detection_result)
    all_bboxes = json.load(f)
    
    # os.chdir('../')
    # Recognize text
    trocr = TrOCRRunner(opt)
    all_labels = trocr.run(all_bboxes)

    # Visualize
    save_result(opt, all_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # yolov7 config
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--detection_result', type=str, default='', help='source')

    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--font', type=str, default='', help='font path')
    parser.add_argument('--recog_model', type=str, default='', help='Recognition model path')

    opt = parser.parse_args()
    main(opt)