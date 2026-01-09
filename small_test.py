import os
import cv2
import numpy as np
from pathlib import Path
from time import time

# from emotiefflib.facial_analysis import get_model_list, EmotiEffLibRecognizer

# from EEFRecognizer import EmotiEffLibRecognizerOnnxCustom
from EEFRecognizer_v2 import EmotiEffLibRecognizerOnnxCustom
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

# Папка с изображениями
input_dir = "positive"
# Берём первые три файла
files = sorted(os.listdir(input_dir))[:3]
fer = EmotiEffLibRecognizerOnnxCustom(model_name='f', model_path = 'fp16_fixed_emotion.onnx')
emotions = []

for fname in files:
    input_file = os.path.join(input_dir, fname)

    # Читаем кадр
    frame_bgr = cv2.imread(input_file)
    if frame_bgr is None:
        print(f"Не удалось прочитать файл: {input_file}")
        continue
    # Преобразуем BGR → RGB
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # ---- здесь вы должны вырезать лицо face_img ----
    # Пример: пока просто используем весь кадр как face_img
    face_img = frame
    # Предсказание эмоции (ваш объект fer)
    result = fer.predict_emotions(face_img, logits=True)
    print(result)
    # Визуализация
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(face_img)
    plt.show()
