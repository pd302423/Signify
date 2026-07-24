"""
ONNX Model Export Script for Signify.

Exports trained PyTorch state_dict models to ONNX format for accelerated browser & backend deployment.
"""

import os
import argparse
import torch
from model import DynamicLandmarkLSTM
from cslr_model import CSLR_BiLSTM_CTC


def export_dynamic_to_onnx(model_path: str, output_onnx: str, seq_len: int = 30, feature_dim: int = 225, num_classes: int = 50):
    model = DynamicLandmarkLSTM(input_dim=feature_dim, num_classes=num_classes)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Notice: Checkpoint {model_path} not found. Exporting initialized model.")

    model.eval()
    dummy_input = torch.randn(1, seq_len, feature_dim, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['landmarks_sequence'],
            output_names=['class_probabilities'],
            dynamic_axes={
                'landmarks_sequence': {0: 'batch_size', 1: 'sequence_length'},
                'class_probabilities': {0: 'batch_size'}
            }
        )
        print(f"✅ Successfully exported Dynamic LSTM ONNX model to: {output_onnx}")
    except Exception as e:
        print(f"⚠️ ONNX Export Notice: {e}")


def export_cslr_to_onnx(model_path: str, output_onnx: str, seq_len: int = 90, feature_dim: int = 225, num_glosses: int = 60):
    model = CSLR_BiLSTM_CTC(input_dim=feature_dim, num_glosses=num_glosses)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Notice: Checkpoint {model_path} not found. Exporting initialized CSLR model.")

    model.eval()
    dummy_input = torch.randn(1, seq_len, feature_dim, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['continuous_landmarks'],
            output_names=['logits'],
            dynamic_axes={
                'continuous_landmarks': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            dynamo=False
        )
        print(f"✅ Successfully exported CSLR BiLSTM ONNX model to: {output_onnx}")
    except Exception as e:
        print(f"⚠️ ONNX CSLR Export Notice: {e}")



def main():
    parser = argparse.ArgumentParser(description="Export Signify PyTorch Model to ONNX format")
    parser.add_argument("--input-model", type=str, default="weights/sign_lstm.pth")
    parser.add_argument("--output-onnx", type=str, default="weights/sign_lstm.onnx")
    parser.add_argument("--cslr-onnx", type=str, default="weights/cslr_bilstm.onnx")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_onnx) or ".", exist_ok=True)
    export_dynamic_to_onnx(args.input_model, args.output_onnx)
    export_cslr_to_onnx("weights/cslr_bilstm.pth", args.cslr_onnx)


if __name__ == "__main__":
    main()

