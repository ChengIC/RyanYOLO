import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size,non_max_suppression,scale_coords,xyxy2xywh,increment_path
from utils.datasets import LoadImages
from utils.plots import Annotator, colors, save_one_box
from pathlib import Path
import cv2
import sys
import os 


def calBack(xywh):
    xmin = xywh[0] - xywh[2]/2
    xmax = xywh[0] + xywh[2]/2
    ymin = xywh[1] - xywh[3]/2
    ymax = xywh[1] + xywh[3]/2
    back = [xmin,ymin,xmax,ymax]
    return back

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
name = 'exp'
project = ROOT / 'runs/detect'
exist_ok = False
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# model paras
device = 'cpu'
weights = 'yolov5s.pt'
dnn = 'False'                   # onnx implementation
data='./data/coco128.yaml'      # data labels setting
imgsz=(640, 640)                # inference size
half=False                      # precision
augment=False                   # augmented inference
visualize=False                 # visualize features

# NMS paras
conf_thres=0.25            # confidence threshold
iou_thres=0.45             # NMS IOU threshold
classes=None               # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False         # class-agnostic NMS
max_det=1000               # maximum detections per image

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Images of Imaegs Folders
img = './image_folders'  # or file, Path, PIL, OpenCV, numpy, list
dataset = LoadImages(img, img_size=imgsz, stride=stride, auto=pt)


# Inference
model.warmup(imgsz=(1, 3, *imgsz), half=half)
for path, im, im0s, vid_cap, s in dataset:
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    # print (im.shape)
    # print (im0s.shape)
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # print (im.shape)
    pred = model(im, augment=augment, visualize=visualize)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    

        ########################### Write results ##############################

        # path settings
        p = Path(p)  # to Path

        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

        # save text paras
        save_txt=True
        save_conf=True
        
        # save image paras
        save_img=True
        save_crop=True
        view_img=True
        hide_labels=False
        hide_conf=False
        line_thickness=1
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # normalize image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop

        # save results
        for *xyxy, conf, cls in reversed(det):
            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img or save_crop or view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        

        # Save results (image with detections)
        im0 = annotator.result()
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
