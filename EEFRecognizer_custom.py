import onnx
import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Tuple, Union
from abc import ABC, abstractmethod


class EmotiEffLibRecognizerBaseCustom(ABC):
    def __init__(self, model_name: str) -> None:
        self.is_mtl = "_mtl" in model_name

        # Классы эмоций
        if "_7" in model_name:
            self.idx_to_emotion_class = {
                0: "Anger",
                1: "Disgust",
                2: "Fear",
                3: "Happiness",
                4: "Neutral",
                5: "Sadness",
                6: "Surprise",
            }
        else:
            self.idx_to_emotion_class = {
                0: "Anger",
                1: "Contempt",
                2: "Disgust",
                3: "Fear",
                4: "Happiness",
                5: "Neutral",
                6: "Sadness",
                7: "Surprise",
            }

        # Параметры нормализации и размер входа
        if "mbf_" in model_name:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.img_size = 112
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

            if "_b2_" in model_name:
                self.img_size = 260
            elif "ddamfnet" in model_name:
                self.img_size = 112
            else:
                self.img_size = 224

    @abstractmethod
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def extract_logits(self, face_img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        raise NotImplementedError

    def classify_emotions(self, logits: np.ndarray, apply_softmax: bool = False):
        """
        logits: (N, num_classes)
        """
        # Если модель MTL, последние два выхода — не эмоции
        if self.is_mtl:
            x = logits[:, :-2]
        else:
            x = logits

        preds = np.argmax(x, axis=1)

        if apply_softmax:
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            probs = e_x / e_x.sum(axis=1, keepdims=True)
        else:
            probs = x

        return [self.idx_to_emotion_class[p] for p in preds], probs

    def predict_emotions(self, face_img, logits=True):
        raw_logits = self.extract_logits(face_img)
        return self.classify_emotions(raw_logits, apply_softmax=not logits)


class EmotiEffLibRecognizerOnnxCustom(EmotiEffLibRecognizerBaseCustom):
    """
    Упрощённая версия: модель остаётся целой, без удаления GEMM.
    """

    def __init__(self, model_name: str = "enet_b0_8_best_vgaf", model_path: str = "./model.onnx") -> None:
        super().__init__(model_name)

        # Загружаем модель как есть
        self.ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        # Определяем имя входа и выхода
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        x = cv2.resize(img, (self.img_size, self.img_size)) / 255.0
        for i in range(3):
            x[..., i] = (x[..., i] - self.mean[i]) / self.std[i]
        return x.transpose(2, 0, 1).astype("float32")[np.newaxis, :]

    def extract_logits(self, face_img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        # Один кадр
        if isinstance(face_img, np.ndarray):
            img_tensor = self._preprocess(face_img)

        # Батч кадров
        elif isinstance(face_img, list) and all(isinstance(i, np.ndarray) for i in face_img):
            img_tensor = np.concatenate([self._preprocess(img) for img in face_img], axis=0)
        else:
            raise TypeError("Expected np.ndarray or List[np.ndarray]")

        # Полный прогон модели: сразу получаем logits эмоций
        logits = self.ort_session.run([self.output_name], {self.input_name: img_tensor})[0]
        return logits
