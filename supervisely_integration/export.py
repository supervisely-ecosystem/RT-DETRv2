import os
import argparse
from rtdetrv2_pytorch.tools.export_onnx import main
import subprocess


def export_onnx(checkpoint_path: str, config_path: str, output_dir: str = None):
    output_onnx_path = _resolve_output_path(checkpoint_path, '.onnx', output_dir)
    if os.path.exists(output_onnx_path):
        return output_onnx_path
    args = _get_args_onnx()
    args.config = str(config_path)
    args.resume = str(checkpoint_path)
    args.output_file = str(output_onnx_path)
    args.check = True
    main(args)
    return output_onnx_path


def _get_args_onnx():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    args = parser.parse_args([])
    return args


def export_tensorrt(onnx_path: str, output_dir: str = None, fp16=True):
    output_engine_path = _resolve_output_path(onnx_path, '.engine', output_dir)
    if os.path.exists(output_engine_path):
        return output_engine_path
    # export dynamic shape
    cmd_list = [
        'trtexec',
        '--onnx=' + onnx_path,
        '--saveEngine=' + output_engine_path,
    ]
    if fp16:
        cmd_list.append('--fp16')
    p = subprocess.run(cmd_list)
    return output_engine_path


def _resolve_output_path(checkpoint_path: str, ext: str, output_dir: str = None):
    ext = ext if ext.startswith('.') else '.' + ext
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    checkpoint_name = checkpoint_name + ext
    if output_dir is None:
        output_path = checkpoint_name
    else:
        output_path = os.path.join(output_dir, checkpoint_name)
    return output_path