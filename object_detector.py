import torch
import random
from PIL import Image
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box
# Define colors for different classes




class YOLOv5_detector:
    def __init__(self, img_path, model_weights='yolov5l.pt', imgsize=640, device='cuda'):
        self.img_path = img_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(model_weights, map_location=self.device)
        self.imgsize = check_img_size(imgsize, s=self.model.stride.max())
        self.dataset = LoadImages(img_path, img_size=self.imgsize)
        self.half = device != 'cpu'
        self.stride = int(self.model.stride.max())  # model stride
        if self.half:
            self.model.half()  # to FP16
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.model.names))]



    def detect_objects(self, conf_thres=0.25, iou_thres=0.45, classes=None):
        try:
            assert 0 <= conf_thres <= 1, "confidence threshold must be between 0 and 1"
            assert 0 <= iou_thres <= 1, "IOU threshold must be between 0 and 1"

            results = []
            for path, img, im0s, vid_cap in self.dataset:
                try:
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

                        # Append results
                        results.append(det)
                    else:
                        # Handle case with no detections
                        results.append([])

                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    results.append(None)  # Record failure

            # Combine results and process
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.model.names))]
            final_result = self.annotate_detections(results)
            return final_result

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def annotate_detections(self, detections):
        annotated_images = []
        for idx, detection in enumerate(detections):
            if detection is None:
                continue

            img = self.dataset[idx][2]  # Get original image
            for *xyxy, conf, cls in reversed(detection):
                label = f'{self.model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)], line_thickness=3)

            annotated_images.append(img)

        return annotated_images

