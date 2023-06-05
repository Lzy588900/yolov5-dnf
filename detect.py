import time

import numpy as np
import torch
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import smart_inference_mode



@smart_inference_mode()
def run():
    # Load model
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weights='./weights/yolov5n.pt', device=device, dnn=False, data=False, fp16=True)

    # 读取图片

    im = cv2.imread('data/images/2.jpg')

    im0 = im

    # 处理图片
    im = letterbox(im, (640, 640), stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # 推理
    pred = model(im, augment=False, visualize=False)
    # 非极大值抑制
    pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=None, max_det=1000)

    # 处理推理内容
    for i, det in enumerate(pred):
        # 画框
        annotator = Annotator(im0, line_width=3)
        if len(det):
            # 将转换后的图片画框结果转换成原图上的结果
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                line = cls, *xywh, conf
                annotator.box_label(xyxy, label=str(int(cls)), color=(34, 139, 34), txt_color=(0, 191, 255))
                print(xywh, line)
        im0 = annotator.result()
        cv2.imshow('window', im0)
        cv2.waitKey(0)


if __name__ == "__main__":
    run()
