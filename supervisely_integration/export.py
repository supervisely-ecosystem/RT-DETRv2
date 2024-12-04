import os
import argparse
from rtdetrv2_pytorch.tools.export_onnx import main


def export_onnx(checkpoint_path: str, config_path: str, output_onnx_path: str = None):
    if output_onnx_path is None:
        output_onnx_path = checkpoint_path.replace('.pth', '.onnx')
    if os.path.exists(output_onnx_path):
        return output_onnx_path
    args = _get_args_onnx()
    args.config = str(config_path)
    args.resume = str(checkpoint_path)
    args.file_name = str(output_onnx_path)
    args.check = True
    main(args)
    return output_onnx_path


def export_tensorrt(checkpoint_path: str, config_path: str, output_onnx_path: str = None):
    if output_onnx_path is None:
        output_onnx_path = checkpoint_path.replace('.pth', '.onnx')
    if os.path.exists(output_onnx_path):
        return output_onnx_path
    args = _get_args_tensorrt()
    args.config = str(config_path)
    args.resume = str(checkpoint_path)
    args.file_name = str(output_onnx_path)
    args.check = True
    main(args)
    return output_onnx_path


def _get_args_onnx():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--file-name', '-f', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    args = parser.parse_args([])
    return args


def _get_args_tensorrt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt-file', type=str, required=True)
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    args = parser.parse_args([])
    return args