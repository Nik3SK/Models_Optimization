import cv2
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score

from rknnlite.api import RKNNLite


# ---------------- CONFIG ----------------
RKNN_MODEL = './enet_b0_8_best_afew.rknn'
POSITIVE_DIR = 'positive'
NEGATIVE_DIR = 'negative'
IMG_SIZE = 224
# ---------------------------------------


def load_images_from_folder(folder: str):
    images = []
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    for p in Path(folder).iterdir():
        if p.suffix.lower() in valid_ext:
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    return images


def is_happy(class_idx: int) -> bool:
    # Happiness = 4 (как в твоей idx_to_emotion_class)
    return class_idx == 4


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    ВАЖНО:
    Здесь НЕТ mean/std и /255,
    потому что они уже были заданы в rknn.config()
    """
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)  # NCHW
    img = np.expand_dims(img, axis=0)
    return img


def main():
    # --- load images ---
    print('Загрузка изображений...')
    positive_images = load_images_from_folder(POSITIVE_DIR)
    negative_images = load_images_from_folder(NEGATIVE_DIR)

    all_images = positive_images + negative_images
    true_labels = [1] * len(positive_images) + [0] * len(negative_images)

    print(f'Загружено {len(positive_images)} позитивных и {len(negative_images)} негативных')

    # --- init RKNN ---
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(RKNN_MODEL)
    rknn.init_runtime(
        target='rk3588',
        core_mask=RKNNLite.NPU_CORE_AUTO
    )

    predicted_labels = []
    processing_times = []

    print('Инференс на NPU...')
    for img in tqdm(all_images, unit='img'):
        input_tensor = preprocess(img)

        start = time()
        outputs = rknn.inference(inputs=[input_tensor])
        end = time()

        logits = outputs[0]  # shape: (1, num_classes)
        pred_class = int(np.argmax(logits, axis=1)[0])

        pred = 1 if is_happy(pred_class) else 0

        predicted_labels.append(pred)
        processing_times.append(end - start)

    rknn.release()

    # --- metrics ---
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)

    times = np.array(processing_times)

    print('\n' + '=' * 50)
    print('РЕЗУЛЬТАТЫ RKNN (NPU, OrangePi 5)')
    print('=' * 50)
    print(f'Accuracy : {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print()
    print('ВРЕМЯ ИНФЕРЕНСА')
    print('=' * 50)
    print(f'Среднее: {times.mean():.6f} сек')
    print(f'Минимум: {times.min():.6f} сек')
    print(f'Максимум: {times.max():.6f} сек')
    print(f'FPS (mean): {1.0 / times.mean():.2f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
