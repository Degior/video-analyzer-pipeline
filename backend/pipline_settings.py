from typing import Optional, Literal

from pydantic_settings  import BaseSettings


class PipelineSettings(BaseSettings):
    enable_stt: bool = False
    whisper_model_size: Optional[Literal["tiny", "base", "small"]] = None
    enable_correction: bool = False
    enable_summarization: bool = False

    enable_yolo: bool = False
    yolo_size: Optional[Literal["nano", "small", "medium"]] = None

    enable_objects: bool = False
    ocr_models: Optional[Literal["EasyOCR", "TrOCR"]] = None
    yolo_size_ocr: Optional[Literal["без", "nano", "small", "medium"]] = None

    enable_classification: bool = False
    class_models: Optional[Literal["LogReg", "MoE", "DBoF"]] = None

    class Config:
        env_prefix = 'PIPELINE_'
