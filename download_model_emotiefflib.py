from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
print(get_model_list())
fer = EmotiEffLibRecognizer(engine="onnx", model_name='enet_b0_8_va_mtl', device='cpu')