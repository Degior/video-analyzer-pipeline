import json
import torch
import numpy as np
from PIL import Image
from torchvision import models
from backend.transforms import transform_classification
from models.classifiers.log_reg import LogReg
from models.classifiers.mixture_of_experts import MixtureOfExperts
from models.classifiers.dbof import DBoF


class VideoClassifier:
    def __init__(self, device=None,
                 model_name="LogReg",
                 logreg_path="models/classification/logreg.pth",
                 moe_path="models/classification/moe.pth",
                 dbof_path="models/classification/dbof.pth",
                 labels_path="models/classification/classification_labels.json",
                 ):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if model_name == "LogReg":
            self.logreg = LogReg(input_dim=512, num_classes=104).to(self.device)
            self.logreg.linear.load_state_dict(torch.load(logreg_path, map_location=self.device))
            self.logreg.eval()

        elif model_name == "MixtureOfExperts":
            self.moe = MixtureOfExperts(input_dim=512, num_classes=104).to(self.device)
            self.moe.load_state_dict(torch.load(moe_path, map_location=self.device))
            self.moe.eval()

        else:
            self.dbof = DBoF(input_dim=512, projection_dim=256, num_classes=104, pooling='max').to(self.device)
            self.dbof.load_state_dict(torch.load(dbof_path, map_location=self.device))
            self.dbof.eval()


        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.feature_extractor.eval()

        with open(labels_path, "r", encoding="utf-8") as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}

    def classify(self, frame_paths):
        features = []

        for path in frame_paths:
            image = Image.open(path).convert("RGB")
            tensor = transform_classification(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.feature_extractor(tensor).squeeze().cpu().numpy()
            features.append(embedding)

        video_embedding = np.stack(features, axis=0)

        if self.model_name == "DBoF":
            input_tensor = torch.tensor(video_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            input_tensor = torch.tensor(np.mean(video_embedding, axis=0), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model_name == "LogReg":
                logits = self.logreg(input_tensor)
            elif self.model_name == "MoE":
                logits = self.moe(input_tensor)
            elif self.model_name == "DBoF":
                logits = self.dbof(input_tensor)
            else:
                raise ValueError(f"Неизвестная модель классификации: {self.model_name}")

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        max_conf = np.max(probs)
        predicted_class = np.argmax(probs)
        class_name = self.label_map.get(predicted_class, "Неизвестный класс")

        if max_conf < 0.3:
            return "Пользовательский"
        return class_name
