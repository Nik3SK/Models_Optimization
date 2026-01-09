from rknn.api import RKNN
import cv2
import os
import numpy as np

MODEL_ONNX = 'model.onnx'
MODEL_RKNN = 'model.rknn'
DATASET = './dataset'

rknn = RKNN(verbose=True, verbose_file='rknn_logs.log')

# 1. Конфигурация
rknn.config(
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8',
    optimization_level=3
)

# 2. Загрузка ONNX
ret = rknn.load_onnx(
    model=MODEL_ONNX,
)
assert ret == 0

# 3. Подготовка калибровочного датасета
def dataset_generator():
    for img_name in os.listdir(DATASET):
        img = cv2.imread(os.path.join(DATASET, img_name))
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        img = img[..., ::-1]  # BGR → RGB если нужно
        img = img.astype(np.float32) / 255.0
        yield [img]

# 4. Построение RKNN
ret = rknn.build(
    do_quantization=True,
    dataset=dataset_generator()
)
assert ret == 0

# 5. Экспорт
ret = rknn.export_rknn(MODEL_RKNN)
assert ret == 0

rknn.release()
