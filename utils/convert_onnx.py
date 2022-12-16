import torch
from models import cnn10, crnn10
import onnxruntime as ort
import onnx
import numpy as np

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", default="./onnx")

    args = parser.parse_args()

    if args.model == "CNN10":
        model = cnn10(64, 2, args.model_path)
        dummy_input = torch.randn(1, 1, 32, 64)
        input_names = ["actual_input_1"] + \
            ["learned_%d" % i for i in range(16)]
        output_names = ["output1"]
        output_path = args.output_dir + "/" + "cnn10.onnx"
        torch.onnx.export(model, dummy_input, output_path, verbose=True,
                          input_names=input_names, output_names=output_names, opset_version=11)

        onnx_model = onnx.load(output_path)

        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

        ort_session = ort.InferenceSession(output_path)

        inp_onnx = np.random.randn(1, 1, 32, 64)
        onnx_output = ort_session.run(
            None,
            {"actual_input_1": inp_onnx.astype(np.float32)}
        )
        inp_torch = torch.from_numpy(inp_onnx).float()
        model.eval()
        torch_output = model(inp_torch)
        print(onnx_output)
        print("=============================")
        print(torch_output)
    elif args.model == "CRNN10":
        model = crnn10(64, 2, args.model_path)
        dummy_input = torch.randn(1, 1, 32, 64)
        input_names = ["actual_input_1"] + \
            ["learned_%d" % i for i in range(16)]
        output_names = ["output1"]
        output_path = args.output_dir + "/" + "crnn10.onnx"
        torch.onnx.export(model, dummy_input, output_path, verbose=True,
                          input_names=input_names, output_names=output_names, opset_version=11)
        onnx_model = onnx.load(output_path)

        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

        ort_session = ort.InferenceSession(output_path)

        inp_onnx = np.random.randn(1, 1, 32, 64)
        onnx_output = ort_session.run(
            None,
            {"actual_input_1": inp_onnx.astype(np.float32)}
        )
        inp_torch = torch.from_numpy(inp_onnx).float()
        model.eval()
        torch_output = model(inp_torch)
        print(onnx_output)
        print("=============================")
        print(torch_output)
    elif args.model == "CRNN":
        model = crnn10(64, 2, args.model_path)
        dummy_input = torch.randn(1, 1, 32, 64)
        input_names = ["actual_input_1"] + \
            ["learned_%d" % i for i in range(16)]
        output_names = ["output1"]
        output_path = args.output_dir + "/" + "crnn.onnx"
        torch.onnx.export(model, dummy_input, output_path, verbose=True,
                          input_names=input_names, output_names=output_names, opset_version=11)
        onnx_model = onnx.load(output_path)

        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

        ort_session = ort.InferenceSession(output_path)

        inp_onnx = np.random.randn(1, 1, 32, 64)
        onnx_output = ort_session.run(
            None,
            {"actual_input_1": inp_onnx.astype(np.float32)}
        )
        inp_torch = torch.from_numpy(inp_onnx).float()
        model.eval()
        torch_output = model(inp_torch)
        print(onnx_output)
        print("=============================")
        print(torch_output)
