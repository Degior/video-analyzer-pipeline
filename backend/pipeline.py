import tempfile
import os
from backend.utils import extract_audio, extract_frames


class VideoProcessor:
    def __init__(
            self,
            whisper_size=None,
            correct_t5=False,
            summarize_t5=False,
            enable_yolo=None,
            yolo_size_ocr=None,
            yolo_size=None,
            enable_objects=False,
            ocr_models=None,
            class_models=None
    ):
        self.whisper_size = whisper_size
        self.correct_t5 = correct_t5
        self.summarize_t5 = summarize_t5
        self.enable_yolo = enable_yolo
        self.yolo_size_ocr = yolo_size_ocr
        self.yolo_size = yolo_size
        self.enable_objects = enable_objects
        self.ocr_model = ocr_models
        self.class_model = class_models

    def process(self, video_file):
        results = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            audio_path = os.path.join(tmpdir, "audio.wav")
            extract_audio(video_path, audio_path)
            frames = extract_frames(video_path, tmpdir)

            if self.whisper_size:
                from backend.transcription import WhisperT5Processor
                processor = WhisperT5Processor()
                transcript = processor.transcribe_and_correct(
                    audio_path=audio_path,
                    whisper_size=self.whisper_size,
                    correct=self.correct_t5,
                )
                results["transcript"] = transcript

                if self.summarize_t5:
                    from backend.summarizer import T5Summarizer
                    summarizer = T5Summarizer()
                    results["summary"] = summarizer.summarize(transcript)

            if self.enable_yolo:
                from backend.object_detection import YoloObjectDetector
                object_detector = YoloObjectDetector(model_size=self.yolo_size)
                results["objects"] = object_detector.detect_objects(frames)

            if self.enable_objects:
                from backend.ocr import OCRProcessor
                ocr_processor = OCRProcessor(ocr_model=self.ocr_model, yolo_size=self.yolo_size_ocr)
                results["ocr"] = ocr_processor.extract_text_from_frames(frames)

            if self.class_model:
                from backend.classification import VideoClassifier
                vc = VideoClassifier(model_name=self.class_model)
                results["classification"] = vc.classify(frames)

        return results
