import torch
from torchvision.models.resnet import resnet50


def main() -> None:
    model = resnet50(weights="DEFAULT")

    torch.onnx.export(
        model,
        torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        "resnet50_opset13.onnx",
        opset_version=13,
    )


if __name__ == "__main__":
    main()
