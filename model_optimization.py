import onnx
from onnxconverter_common import float16
from onnx import helper, TensorProto, numpy_helper
import numpy as np
from typing import List, Tuple, Union


INPUT = "./models/enet_b0_8_va_mtl.onnx"
OUTPUT = "./models/fp16_enet_b0_8_va_mtl.onnx"


def convert_to_fp16_preserve_output(model):
    # 1. выполняем стандартную конверсию
    fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    # вход/выход остаются FLOAT32
    # скрытые слои становятся FP16
    return fp16

def main():
    model = onnx.load(INPUT)

    model = convert_to_fp16_preserve_output(model)

    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT)

    print("Сохранено:", OUTPUT)


if __name__ == "__main__":
    main()

