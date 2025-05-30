from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class T5Summarizer:
    def __init__(self, model_path="models/t5_summarizer", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)

    def summarize(self, text, max_length=512, min_length=256, num_beams=4):
        input_text = "summarize: " + text
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary
