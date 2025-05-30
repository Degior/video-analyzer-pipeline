from ultralytics import YOLO
import os

class YoloObjectDetector:
    def __init__(self, model_size="small", models_dir="models/yolo"):
        size_map = {"nano": "n", "small": "s", "medium": "m"}
        suffix = size_map.get(model_size, "s")
        yolo_filename = f"yolo11{suffix}.pt"
        yolo_path = os.path.join(models_dir, yolo_filename)
        self.model = YOLO(yolo_path)

    def detect_objects(self, frame_paths, max_frames=10):
        detected_objects = set()
        for frame_path in frame_paths[:max_frames]:
            results = self.model(frame_path)
            names = results[0].names
            boxes = results[0].boxes
            if boxes is not None:
                for cls_id in boxes.cls.tolist():
                    detected_objects.add(names[int(cls_id)])
        return list(detected_objects)
