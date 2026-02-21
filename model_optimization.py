import onnx
from onnxconverter_common import float16, auto_mixed_precision
from onnx import helper, TensorProto, numpy_helper
import numpy as np
from typing import List, Tuple, Union
import cv2


INPUT = "./models/enet_b0_8_va_mtl.onnx"
OUTPUT = "./models/fp16_mixed_enet_b0_8_va_mtl.onnx"


def convert_to_fp16_preserve_output(model):
    # 1. выполняем стандартную конверсию
    fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    # вход/выход остаются FLOAT32
    # скрытые слои становятся FP16
    return fp16

def convert_to_fp16_mixed(model, feed):
    # Assuming x is the input to the model
    model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model, feed, rtol=0.01, atol=0.001, keep_io_types=True)
    return model_fp16

def preprocess(img_path) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    x = cv2.resize(img, (224, 224)) / 255.0
    mean_data = [0.485, 0.456, 0.406]
    std_data = [0.229, 0.224, 0.225]
    for i in range(3):
        x[..., i] = (x[..., i] - mean_data[i]) / std_data[i]
    return x.transpose(2, 0, 1).astype("float32")[np.newaxis, :]

def main():
    model = onnx.load(INPUT)
    example_image = 'example_image.jpg'
    feed = {'input': preprocess(example_image)}
    model = convert_to_fp16_mixed(model, feed)
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT)

    print("Сохранено:", OUTPUT)


if __name__ == "__main__":
    main()

