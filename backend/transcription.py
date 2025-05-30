from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import librosa
import os

class WhisperT5Processor:
    def __init__(
        self,
        whisper_model_dir="models/whisper",
        t5_corrector_path="models/t5_corrector",
        language="russian",
        device=None
    ):
        self.whisper_model_dir = whisper_model_dir
        self.t5_corrector_path = t5_corrector_path
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.t5_tokenizer = AutoTokenizer.from_pretrained(self.t5_corrector_path)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(self.t5_corrector_path).to(self.device)

    def transcribe_and_correct(self, audio_path, whisper_size="small", correct=False, chunk_length=30):
        whisper_path = os.path.join(self.whisper_model_dir, whisper_size)
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_path).to(self.device)
        whisper_processor = AutoProcessor.from_pretrained(whisper_path)

        audio, sr = librosa.load(audio_path, sr=16000)
        chunk_samples = chunk_length * sr
        total_samples = audio.shape[0]

        transcripts = []

        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk_audio = audio[start_sample:end_sample]

            inputs = whisper_processor(chunk_audio, sampling_rate=sr, return_tensors="pt").to(self.device)
            with torch.no_grad():
                predicted_ids = whisper_model.generate(
                    inputs.input_features,
                    forced_decoder_ids=whisper_processor.get_decoder_prompt_ids(
                        language=self.language, task="transcribe"
                    ),
                    no_repeat_ngram_size=2,
                )
            chunk_transcript = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcripts.append(chunk_transcript)

        full_transcript = " ".join(transcripts)

        if not correct:
            return full_transcript

        input_ids = self.t5_tokenizer.encode(
            "fix: " + full_transcript, return_tensors="pt", max_length=2048, truncation=True
        ).to(self.device)

        with torch.no_grad():
            corrected_ids = self.t5_model.generate(input_ids, max_length=512)
        corrected = self.t5_tokenizer.decode(corrected_ids[0], skip_special_tokens=True)

        return corrected

