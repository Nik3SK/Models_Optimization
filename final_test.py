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


def load_images_from_folder(folder: str) -> list[np.ndarray]:
    """Загружает все изображения из папки как numpy массивы (BGR → RGB)."""
    images = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    for file_path in Path(folder).iterdir():
        if file_path.suffix.lower() in valid_extensions:
            img = cv2.imread(str(file_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
    return images


def is_happy(emotion_label: str) -> bool:
    """Определяет, является ли эмоция 'улыбкой' (метка 'happiness')."""
    return emotion_label.lower() == "happiness"


def main():
    # Пути к папкам
    positive_dir = "positive"
    negative_dir = "negative"
    # Инициализация модели
    device = "cpu"
    # model_name = get_model_list()[1]  # Берём первую доступную модель
    fp16 = 'fp16_fixed_emotion.onnx'
    fp32 = 'enet_b0_8_best_afew.onnx'
    int8 = 'model_int8.onnx'
    fer = EmotiEffLibRecognizerOnnxCustom(model_name='d', model_path = fp16)
    # Загрузка изображений
    print("Загрузка изображений...")
    positive_images = load_images_from_folder(positive_dir)
    negative_images = load_images_from_folder(negative_dir)
    print(f"Загружено {len(positive_images)} позитивных и {len(negative_images)} негативных изображений.")
    # Подготовка данных
    all_images = positive_images + negative_images
    true_labels = [1] * len(positive_images) + [0] * len(negative_images)  # 1 = happiness, 0 = not
    predicted_labels = []
    processing_times = []
    # Обработка изображений с прогресс-баром
    print("Обработка изображений...")
    for img in tqdm(all_images, desc="Обработка кропов", unit="изображение"):
        start_time = time()
        try:
            emotion, _ = fer.predict_emotions(img, logits=True)
            pred = 1 if is_happy(emotion[0]) else 0
        except Exception as e:
            # Логировать ошибку можно при желании, но для метрик предсказываем 0
            pred = 0
        end_time = time()
        predicted_labels.append(pred)
        processing_times.append(end_time - start_time)
    # Расчёт метрик
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    # Статистика по времени
    times = np.array(processing_times)
    mean_time = times.mean()
    min_time = times.min()
    max_time = times.max()
    # Вывод результатов
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ emotiefflib")
    print("="*50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print()
    print("СТАТИСТИКА ПО ВРЕМЕНИ ОБРАБОТКИ")
    print("="*50)
    print(f"Среднее время на изображение: {mean_time:.4f} сек")
    print(f"Минимальное время:           {min_time:.4f} сек")
    print(f"Максимальное время:          {max_time:.4f} сек")
    print("="*50)


if __name__ == "__main__":
    main()