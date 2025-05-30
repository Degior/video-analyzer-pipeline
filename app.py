
import streamlit as st
from backend.pipeline import VideoProcessor
from backend.pipline_settings import PipelineSettings


class PipelineUI:
    def __init__(self):
        self.pipeline_settings = PipelineSettings()
        self.video_file = None
        self.results = {}

    def render_sidebar(self):
        st.sidebar.header("🔧 Настройка пайплайна")

        self.pipeline_settings.enable_stt = st.sidebar.checkbox(
            "💬 Распознавание речи (Whisper)", value=self.pipeline_settings.enable_stt)
        if self.pipeline_settings.enable_stt:
            self.pipeline_settings.whisper_model_size = st.sidebar.selectbox(
                "Whisper-модель", ["tiny", "base", "small"], index=0)
            self.pipeline_settings.enable_correction = st.sidebar.checkbox(
                "Коррекция (T5-base)", value=self.pipeline_settings.enable_correction)
            self.pipeline_settings.enable_summarization = st.sidebar.checkbox(
                "Суммаризация (T5-base)", value=self.pipeline_settings.enable_summarization)

        self.pipeline_settings.enable_yolo = st.sidebar.checkbox(
            "🔍 Нахождение объектов (YOLO)", value=self.pipeline_settings.enable_yolo)
        if self.pipeline_settings.enable_yolo:
            self.pipeline_settings.yolo_size = st.sidebar.selectbox(
                "YOLOv11 модель", ["nano", "small", "medium"], index=0)

        self.pipeline_settings.enable_objects = st.sidebar.checkbox(
            "🔣 Поиск текста (OCR)", value=self.pipeline_settings.enable_objects)
        if self.pipeline_settings.enable_objects:
            self.pipeline_settings.ocr_models = st.sidebar.selectbox(
                "OCR модель", ["EasyOCR", "TrOCR"])
            self.pipeline_settings.yolo_size_ocr = st.sidebar.selectbox(
                "YOLOv11 модель", ["без", "nano", "small", "medium"])

        self.pipeline_settings.enable_classification = st.sidebar.checkbox(
            "🏷️ Классификация видео (Основано на UCF101)", value=self.pipeline_settings.enable_classification)
        if self.pipeline_settings.enable_classification:
            self.pipeline_settings.class_models = st.sidebar.selectbox(
                "Классификаторы типа видео", ["LogReg", "MoE", "DBoF"])

    def upload_video(self):
        self.video_file = st.file_uploader("Загрузите видеофайл", type=["mp4", "avi", "mov", "mkv", "webm"])


    def run_analysis(self):
        if self.video_file and st.button("▶️ Запустить анализ"):
            with st.spinner("Обрабатываем видео..."):
                processor = VideoProcessor(
                    whisper_size=self.pipeline_settings.whisper_model_size if self.pipeline_settings.enable_stt else None,
                    correct_t5=self.pipeline_settings.enable_correction if self.pipeline_settings.enable_stt else False,
                    summarize_t5=self.pipeline_settings.enable_summarization if self.pipeline_settings.enable_stt else False,
                    enable_yolo=self.pipeline_settings.enable_yolo,
                    yolo_size_ocr=self.pipeline_settings.yolo_size_ocr if self.pipeline_settings.yolo_size_ocr != "без" else None,
                    yolo_size=self.pipeline_settings.yolo_size if self.pipeline_settings.enable_yolo else None,
                    enable_objects=self.pipeline_settings.enable_objects,
                    ocr_models=self.pipeline_settings.ocr_models if self.pipeline_settings.enable_objects else None,
                    class_models=self.pipeline_settings.class_models if self.pipeline_settings.enable_classification else None,
                )
                self.results = processor.process(self.video_file)

    def show_results(self):
        if self.results:
            st.subheader("📄 Результаты анализа")
        if "transcript" in self.results:
            st.text_area("Транскрипция", self.results["transcript"], height=150)
        if "summary" in self.results:
            st.text_area("Суммаризация", self.results["summary"], height=150)
        if "objects" in self.results:
            st.text_area("На видео нашлись такие предметы как", "\n".join(self.results["objects"]), height=150)
        if "ocr" in self.results:
            st.text_area("Текст в кадре", self.results["ocr"], height=150)
        if "classification" in self.results:
            st.text_area("Классификация видео", self.results["classification"], height=150)


def main():
    st.set_page_config(page_title="Видео-Анализатор", layout="wide")
    st.title("🎥 Анализатор видео с помощью нейросетей")

    ui = PipelineUI()
    ui.render_sidebar()
    ui.upload_video()
    ui.run_analysis()
    ui.show_results()


main()
