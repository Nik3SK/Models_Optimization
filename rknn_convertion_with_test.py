from rknn.api import RKNN
import cv2
import os
import numpy as np

MODEL_ONNX = 'enet_b0_8_best_afew.onnx'
MODEL_RKNN = 'enet_b0_8_best_afew.rknn'
DATASET = './sample_crops_positive'

rknn = RKNN(verbose=True, verbose_file='rknn_logs.log')

# 1. Конфигурация
rknn.config(
    mean_values = [[0.485, 0.456, 0.406]], 
    std_values= [[0.229, 0.224, 0.225]], 
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8',
    optimization_level=3
)

# 2. Загрузка ONNX
ret = rknn.load_onnx(
    model=MODEL_ONNX,
    inputs=['input'],
    input_size_list=[[1, 3, 224, 224]])
assert ret == 0

# 3. Подготовка калибровочного датасета
DATASET_TXT = './dataset.txt'

with open(DATASET_TXT, 'w') as f:
    for img_name in os.listdir(DATASET):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            f.write(os.path.join(DATASET, img_name) + '\n')

# 4. Построение RKNN
ret = rknn.build(
    do_quantization=False,
    dataset='./dataset.txt'
)
assert ret == 0

# 5. Экспорт
ret = rknn.export_rknn(MODEL_RKNN)
assert ret == 0

# 6. Проверка работы модели
rknn.init_runtime()
IMAGE_PATH = './sample_crops_positive/14_00_40_face0.jpg'

def preprocess(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(img, (224,224)) / 255.0
    for i in range(3):
        x[..., i] = (x[..., i] - [0.485, 0.456, 0.406][i]) / [0.229, 0.224, 0.225][i]
    return x.transpose(2, 0, 1).astype("float32")[np.newaxis, :]

input_data = preprocess(IMAGE_PATH)

outputs = rknn.inference(inputs=[input_data], data_format='nchw')

print("Output shape:", outputs[0].shape)
print("Output sample:", outputs[0][0][:10])
print(outputs)

rknn.release()