import torch
from PIL import Image
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.plots import Annotator

class ImageProcessor:
    def __init__(self, img_path, model_weights='yolov5l.pt', imgsz=640):
        self.img_path = img_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(model_weights, map_location=self.device)
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())
        self.dataset = LoadImages(img_path, img_size=self.imgsz)

    def detect_objects(self, conf_thres=0.25, iou_thres=0.45, classes=None):
        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)

            # Process detections
            det = pred[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}"  # add to string

                annotator = Annotator(im0s, line_width=3, example=str(self.model.names))
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=Annotator.colors[int(cls), True])

                im0 = annotator.result()
                return im0