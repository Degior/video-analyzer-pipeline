import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import easyocr
from PIL import Image

class OCRProcessor:
    def __init__(self, ocr_model="TrOCR", yolo_size=None,
                 trocr_path="models/trocr", yolo_models_dir="models/yolo-ocr"):
        self.ocr_model = ocr_model
        self.yolo_size = yolo_size
        self.yolo_model = None
        self.yolo_models_dir = yolo_models_dir

        if self.ocr_model == "TrOCR":
            self.trocr_processor = TrOCRProcessor.from_pretrained(trocr_path)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_path)
        elif self.ocr_model == "EasyOCR":
            self.easy_reader = easyocr.Reader(["ru", "en"], download_enabled=True)

        if self.yolo_size is not None:
            yolo_filename = f"yolo{self.yolo_size[0]}.pt"
            yolo_path = os.path.join(self.yolo_models_dir, yolo_filename)
            self.yolo_model = YOLO(yolo_path)

    def extract_text_from_frames(self, frame_paths):
        texts = []
        seen_texts = set()

        for path in frame_paths[:10]:
            if self.yolo_model:
                results = self.yolo_model(path)
                has_text = any(
                    "text" in result.names or "Text" in result.names or
                    "0" in result.names or 0 in result.names
                    for result in results
                )
                if not has_text:
                    continue

            frame_texts = []

            if self.ocr_model == "EasyOCR":
                img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                result_easy = self.easy_reader.readtext(img_gray, detail=0)
                easy_text = ' '.join(result_easy)
                if easy_text and easy_text not in seen_texts:
                    seen_texts.add(easy_text)
                    frame_texts.append(easy_text)

            elif self.ocr_model == "TrOCR":
                image = Image.open(path).convert("RGB")
                pixel_values = self.trocr_processor(images=image, return_tensors="pt").pixel_values
                generated_ids = self.trocr_model.generate(pixel_values)
                trocr_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                if trocr_text and trocr_text not in seen_texts:
                    seen_texts.add(trocr_text)
                    frame_texts.append(trocr_text)

            if frame_texts:
                texts.append("\n".join(frame_texts))

        return "\n\n".join(texts)
