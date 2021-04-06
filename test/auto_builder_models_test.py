from models.auto_builder_models import (
    EasyPeasyResNet,
    EasyPeasyConvNet,
    EasyPeasyConvRelationalNet,
)

import torch

RUN_CUDA_TESTS = False


def apply_to_test_device(model, input_tensor):
    if torch.cuda.is_available() and RUN_CUDA_TESTS:
        model = model.to(torch.cuda.current_device())

        input_tensor = input_tensor.to(torch.cuda.current_device())

    else:

        model = model.to(torch.device("cpu"))

        input_tensor = input_tensor.to(torch.device("cpu"))

    return model, input_tensor


def test_EasyPeasyConvNet_layer_output_shape():
    model = EasyPeasyConvNet(
        num_classes=10,
        kernel_size=3,
        filter_list=[16, 8, 64],
        stride=1,
        padding=1,
    )
    dummy_x = torch.zeros((8, 3, 128, 128))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert out.shape[1] == 10
    assert len(out.shape) == 2


def test_EasyPeasyConvRelationalNet_layer_output_shape():
    model = EasyPeasyConvRelationalNet(
        num_classes=10,
        kernel_size=3,
        filter_list=[16, 8, 64],
        stride=1,
        padding=1,
        relational_num_filters=64,
        relational_num_outputs=64,
        relational_num_layers=3,
    )
    dummy_x = torch.zeros((8, 3, 128, 128))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert out.shape[1] == 10
    assert len(out.shape) == 2


def test_EasyPeasyResNet_layer_output_shape():
    model = EasyPeasyResNet(
        num_classes=10,
        model_name_to_download="resnet18",
        pretrained=True,
    )
    dummy_x = torch.zeros((8, 3, 128, 128))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert out.shape[1] == 10
    assert len(out.shape) == 2

    model = EasyPeasyResNet(
        num_classes=10,
        model_name_to_download="resnet18",
        pretrained=False,
    )
    dummy_x = torch.zeros((8, 3, 128, 128))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert out.shape[1] == 10
    assert len(out.shape) == 2
