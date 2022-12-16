import onnxruntime as ort
import onnx
import numpy as np
from models import crnn

model = onnx.load("onnx/crnn.onnx")

onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))


ort_session = ort.InferenceSession("onnx/crnn.onnx")

outputs = ort_session.run(
    None,
    {"actual_input_1": np.random.randn(32, 1, 500, 64).astype(np.float32)}
)

print(outputs[0])
print(outputs[1])
