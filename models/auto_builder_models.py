from __future__ import print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print


class ClassificationModel(nn.Module):
    def __init__(
        self,
        feature_embedding_module_list,
        feature_embedding_args,
        num_classes,
    ):
        self.feature_embedding_module_list = feature_embedding_module_list
        self.feature_embedding_args = feature_embedding_args
        self.num_classes = num_classes
        self.is_layer_built = False
        super(ClassificationModel, self).__init__()

    def build(self, input_shape):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        out = torch.zeros(input_shape)
        print("Building basic block of a classification model using input shape")
        # assumes that input shape is b, c, h, w
        b, c, h, w = out.shape
        print("build input", out.shape)

        if isinstance(self.feature_embedding_module_list, list):
            self.feature_embedding_module = nn.Sequential(
                OrderedDict(
                    [
                        (str(idx), item(**item_args))
                        for idx, (item, item_args) in enumerate(
                            zip(
                                self.feature_embedding_module_list,
                                self.feature_embedding_args,
                            )
                        )
                    ]
                )
            )
        else:
            self.feature_embedding_module = self.feature_embedding_module_list(
                **self.feature_embedding_args
            )

        out = self.feature_embedding_module.forward(out)
        print("build input", out.shape)

        out = out.view(out.shape[0], -1)
        print("build input", out.shape)

        # classification head
        self.output_layer = nn.Linear(
            in_features=out.shape[1], out_features=self.num_classes, bias=True
        )
        out = self.output_layer.forward(out)
        print("build input", out.shape)

        self.is_layer_built = True
        print("Summary:")
        print(
            "Build ",
            self.__class__.__name__,
            "with input shape",
            input_shape,
            "with output shape",
            out.shape,
        )
        for layer_idx, layer_params in self.named_parameters():
            print(layer_idx, layer_params.shape)

    def forward(self, x):
        """
        Forward propagates the network given an input batch
        :param x: Inputs x (b, c, w, h)
        :return: preds (b, num_classes)
        """

        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x
        features = self.feature_embedding_module.forward(out)
        out = features.view(features.shape[0], -1)
        out = self.output_layer.forward(out)
        return out, features


class ResNet(nn.Module):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
    https://github.com/pytorch/vision/blob/331f126855a106d29e1de23f8bbc3cde66c603e5/
    torchvision/models/resnet.py#L144
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    """

    def __init__(self, model_name_to_download, pretrained=True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.model_name_to_download = model_name_to_download
        self.is_layer_built = False

    def build(self, input_shape):
        x_dummy = torch.zeros(input_shape)
        out = x_dummy

        self.linear_transform = nn.Conv2d(
            in_channels=out.shape[1],
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )

        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0",
            self.model_name_to_download,
            pretrained=self.pretrained,
        )

        self.model.fc = nn.Identity()

        self.model.avgpool = nn.Identity()

        out = self.linear_transform.forward(out)

        out = self.model.forward(out)

        out = out.view(out.shape[0], 512, -1)

        out = out.view(
            out.shape[0], 512, int(np.sqrt(out.shape[-1])), int(np.sqrt(out.shape[-1]))
        )

        self.is_layer_built = True

        print(
            "Build ",
            self.__class__.__name__,
            "with input shape",
            input_shape,
            "with output shape",
            out.shape,
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        out = self.linear_transform.forward(out)
        # out shape (b*s, channels)
        out = self.model.forward(out)

        out = out.view(out.shape[0], 512, -1)

        out = out.view(
            out.shape[0], 512, int(np.sqrt(out.shape[-1])), int(np.sqrt(out.shape[-1]))
        )

        return out


class BatchRelationalModule(nn.Module):
    def __init__(
        self,
        num_filters,
        num_layers,
        num_outputs,
        bias,
        num_post_processing_filters,
        num_post_processing_layers,
        avg_pool_input_shape=None,
    ):
        super(BatchRelationalModule, self).__init__()

        self.block_dict = nn.ModuleDict()
        self.num_post_processing_filters = num_post_processing_filters
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.bias = bias
        self.avg_pool_input_shape = avg_pool_input_shape
        self.num_post_processing_layers = num_post_processing_layers
        self.layer_is_built = False

    def build(self, input_shape):
        out_img = torch.zeros(input_shape)
        """g"""

        if len(out_img.shape) == 4:
            b, c, h, w = out_img.shape
            if self.avg_pool_input_shape is not None:
                out_img = F.adaptive_avg_pool2d(
                    out_img, output_size=self.avg_pool_input_shape
                )
            out_img = out_img.view(b, c, h * w)

        elif len(out_img.shape) == 3:
            b, c, h = out_img.shape  # c are the depth/features/channels

            if self.avg_pool_input_shape is not None:
                out_img = F.adaptive_max_pool1d(
                    out_img, output_size=self.avg_pool_input_shape
                )

            # print(out_img.shape)

        elif len(out_img.shape) == 5:
            b, s, c, h, w = out_img.shape
            # print(out_img.shape)
            out_img = out_img.view(b, s, -1)

            out_img = out_img.permute([0, 2, 1])

            if self.avg_pool_input_shape is not None:
                out_img = F.adaptive_max_pool1d(
                    out_img, output_size=self.avg_pool_input_shape
                )

        out_img = out_img.permute([0, 2, 1])  # b, h*w, c or (b, h, c)
        (
            b,
            length,
            c,
        ) = (
            out_img.shape
        )  # length is the features/slices that will be applied relational module.
        # print(out_img.shape)
        # x_flat = (64 x 25 x 24)
        self.coord_tensor = []
        for i in range(length):
            self.coord_tensor.append(torch.Tensor(np.array([i])))  # 0, 1, ... length-1

        self.coord_tensor = torch.stack(self.coord_tensor, dim=0).unsqueeze(
            0
        )  # 1, length, 1

        if self.coord_tensor.shape[0] != out_img.shape[0]:
            self.coord_tensor = (
                self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])
            )  # batch size, length, 1

        out_img = torch.cat(
            [out_img, self.coord_tensor], dim=2
        )  # batch size, length, c+1

        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)# batch size, 1, length, c+1
        x_i = x_i.repeat(
            1, length, 1, 1
        )  # (h*wxh*wxc)# batch size, length, length, c+1
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc) # batch size, length, 1, c+1
        x_j = x_j.repeat(
            1, 1, length, 1
        )  # (h*wxh*wxc)# batch size, length, length, c+1

        # concatenate all together
        per_location_feature = torch.cat(
            [x_i, x_j], 3
        )  # (h*wxh*wx2*c) # batch size, length, length, (c + 1)*2

        out = per_location_feature.view(
            per_location_feature.shape[0]
            * per_location_feature.shape[1]
            * per_location_feature.shape[2],
            per_location_feature.shape[3],
        )  # batch size * length * length, (c+1)*2
        # print(out.shape)
        for idx_layer in range(self.num_layers):
            self.block_dict["g_fcc_{}".format(idx_layer)] = nn.Linear(
                out.shape[1], out_features=self.num_post_processing_filters
            )
            out = F.leaky_relu(
                self.block_dict["g_fcc_{}".format(idx_layer)].forward(out)
            )  # g_fcc_0= ((c+1)*2, self.num_filters), g_fcc_1 = (self.num_filters,
            # self.num_filters), g_fcc_2 = (self.num_filters, self.num_filters)

        # reshape again and sum
        # print(out.shape)
        out = out.view(
            per_location_feature.shape[0],
            per_location_feature.shape[1],
            per_location_feature.shape[2],
            -1,
        )  # batch size, length, length, self.num_filters
        out = out.mean(1).mean(1)  # batch size, self.num_filters
        # print('here', out.shape)
        """f"""
        for i in range(self.num_post_processing_layers):
            self.block_dict["post_processing_layer_{}".format(i)] = nn.Linear(
                in_features=out.shape[1],
                out_features=self.num_post_processing_filters,
                bias=self.bias,
            )  # (self.num_filters, self.num_filters)
            out = self.block_dict["post_processing_layer_{}".format(i)].forward(out)
            out = F.leaky_relu(out)

        self.output_layer = nn.Linear(
            in_features=out.shape[1], out_features=self.num_outputs, bias=self.bias
        )  # (self.num_filters, self.num_filters)
        out = self.output_layer.forward(out)
        out = F.leaky_relu(out)
        print("Block built with output volume shape", out.shape)

    def forward(self, x_img):

        if not self.layer_is_built:
            self.build(x_img.shape)
            self.layer_is_built = True  # Wenwen: add this line.

        out_img = x_img
        # print("input", out_img.shape)
        """g"""
        if len(out_img.shape) == 4:
            b, c, h, w = out_img.shape
            if self.avg_pool_input_shape is not None:
                out_img = F.adaptive_avg_pool2d(
                    out_img, output_size=self.avg_pool_input_shape
                )
            # print(out_img.shape)
            out_img = out_img.view(b, c, h * w)
        elif len(out_img.shape) == 3:
            b, c, h = out_img.shape

            if self.avg_pool_input_shape is not None:
                out_img = F.adaptive_max_pool1d(
                    out_img, output_size=self.avg_pool_input_shape
                )

            # print(out_img.shape)
            out_img = out_img.view(b, c, h)  # batch size, slices, channels
        elif len(out_img.shape) == 5:
            b, s, c, h, w = out_img.shape
            # print(out_img.shape)
            out_img = out_img.view(b, s, -1)

            out_img = out_img.permute([0, 2, 1])

            if self.avg_pool_input_shape is not None:
                out_img = F.adaptive_max_pool1d(
                    out_img, output_size=self.avg_pool_input_shape
                )

        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape

        if self.coord_tensor.shape[0] != out_img.shape[0]:
            self.coord_tensor = (
                self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])
            )

        # print(out_img.shape, x_img.shape, self.coord_tensor.shape)
        out_img = torch.cat([out_img, self.coord_tensor.to(x_img.device)], dim=2)
        # x_flat = (64 x 25 x 24)
        # print('out_img', out_img.shape)
        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)
        out = per_location_feature.view(
            per_location_feature.shape[0]
            * per_location_feature.shape[1]
            * per_location_feature.shape[2],
            per_location_feature.shape[3],
        )
        # print('large part', out.shape)
        for idx_layer in range(self.num_layers):
            out = F.leaky_relu(
                self.block_dict["g_fcc_{}".format(idx_layer)].forward(out)
            )

        # reshape again and sum
        # print(out.shape)
        out = out.view(
            per_location_feature.shape[0],
            per_location_feature.shape[1],
            per_location_feature.shape[2],
            -1,
        )
        out = out.mean(1).mean(1)
        # print("relational out max {} min{} before FC".format(out.max(), out.min()))

        """f"""
        for i in range(self.num_post_processing_layers):
            out = self.block_dict["post_processing_layer_{}".format(i)].forward(out)
            out = F.leaky_relu(out)
        out = self.output_layer.forward(out)
        out = F.leaky_relu(out)
        return out


class BaseStyleLayer(nn.Module):
    def __init__(self):
        super(BaseStyleLayer, self).__init__()
        self.is_layer_built = False

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        self.is_layer_built = True

        print(
            "Build ",
            self.__class__.__name__,
            "with input shape",
            input_shape,
            "with output shape",
            out.shape,
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self, num_filters, bias):
        super(FullyConnectedLayer, self).__init__()
        self.num_filters = num_filters
        self.bias = bias
        self.is_layer_built = False

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        self.fc1 = nn.Linear(
            in_features=int(np.prod(out.shape[1:])),
            out_features=self.num_filters,
            bias=self.bias,
        )
        out = self.fc1.forward(out)

        self.is_layer_built = True

        print(
            "Build ",
            self.__class__.__name__,
            "with input shape",
            input_shape,
            "with output shape",
            out.shape,
        )

    def forward(self, x):
        assert (
            len(x.shape) == 2
        ), "input tensor should have a length of 2 but its instead {}".format(x.shape)

        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        x = self.fc1(x)

        return x


class Conv2dBNLeakyReLU(nn.Module):
    def __init__(
        self, out_channels, kernel_size, stride, padding, dilation=1, bias=False
    ):
        super(Conv2dBNLeakyReLU, self).__init__()
        self.is_layer_built = False
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        self.conv = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=self.bias,
            padding_mode="zeros",
        )

        out = self.conv.forward(out)

        self.bn = nn.BatchNorm2d(
            track_running_stats=True, affine=True, num_features=out.shape[1], eps=1e-5
        )

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        self.is_layer_built = True

        print(
            "Build ",
            self.__class__.__name__,
            "with input shape",
            input_shape,
            "with output shape",
            out.shape,
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        out = self.conv.forward(out)

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        return out


class Conv2dEmbedding(nn.Module):
    def __init__(
        self,
        layer_filter_list,
        kernel_size,
        stride,
        padding,
        avg_pool_kernel_size=2,
        avg_pool_stride=2,
        dilation=1,
        bias=False,
    ):
        """
        General method details
        :param num_layers_list:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param bias:
        """
        super(Conv2dEmbedding, self).__init__()
        self.is_layer_built = False
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.layer_filter_list = layer_filter_list
        self.avg_pool_kernel_size = avg_pool_kernel_size
        self.avg_pool_stride = avg_pool_stride

    def build(self, input_shape):
        out = torch.zeros(input_shape)
        self.layer_dict = nn.ModuleDict()

        for idx, num_filters in enumerate(self.layer_filter_list):
            self.layer_dict["conv_{}".format(idx)] = Conv2dBNLeakyReLU(
                out_channels=num_filters,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=self.bias,
            )
            out = self.layer_dict["conv_{}".format(idx)].forward(out)

            out = F.avg_pool2d(
                input=out,
                kernel_size=self.avg_pool_kernel_size,
                stride=self.avg_pool_stride,
            )  # TODO: avg_pool kernel_size, stride should be set as parameters

        self.is_layer_built = True

        print(
            "Build ",
            self.__class__.__name__,
            "with input shape",
            input_shape,
            "with output shape",
            out.shape,
        )
        for layer_idx, layer_params in self.named_parameters():
            print(layer_idx, layer_params.shape)

    def forward(self, x):

        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        for idx, num_filters in enumerate(self.layer_filter_list):
            out = self.layer_dict["conv_{}".format(idx)].forward(out)
            out = F.avg_pool2d(
                input=out,
                kernel_size=self.avg_pool_kernel_size,
                stride=self.avg_pool_stride,
            )

        return out


class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    def forward(self, x: list, dim: int):
        x = torch.cat(tensors=x, dim=dim)
        return x


class AvgPoolSpatialAndSliceIntegrator(nn.Module):
    def __init__(self):
        super(AvgPoolSpatialAndSliceIntegrator, self).__init__()
        self.is_layer_built = False

    def build(self, input_shape):
        out = torch.zeros(input_shape)
        # b, s, out_features, w / n, h / n
        out = out.mean(dim=4)
        # b, s, out_features, w / n
        out = out.mean(dim=3)
        # b, s, out_features
        out = out.mean(dim=1)
        # b, out_features

        self.is_layer_built = True

        print(
            "Build ",
            self.__class__.__name__,
            "with input shape",
            input_shape,
            "with output shape",
            out.shape,
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x
        # b, s, out_features, w / n, h / n
        out = out.mean(dim=4)
        # b, s, out_features, w / n
        out = out.mean(dim=3)
        # b, s, out_features
        out = out.mean(dim=1)
        # b, out_features

        return out


class AvgPoolFlexibleDimension(nn.Module):
    def __init__(self, dim):
        super(AvgPoolFlexibleDimension, self).__init__()
        self.is_layer_built = False
        self.dim = dim

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        assert len(input_shape) >= self.dim, (
            "Length of shape is smaller than {}, please ensure the tensor "
            "shape is larger or equal in length".format(self.dim_to_pool_over)
        )

        out = out.mean(dim=self.dim)

        self.is_layer_built = True

        print(
            "Build ",
            self.__class__.__name__,
            "with input shape",
            input_shape,
            "with output shape",
            out.shape,
        )

    def forward(self, x):
        assert len(x.shape) >= self.dim, (
            "Length of shape is smaller than {}, please ensure the tensor "
            "shape is larger or equal in length".format(self.dim_to_pool_over)
        )

        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        out = out.mean(dim=self.dim)

        return out


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(self.dim)


class EasyPeasyResNet(ClassificationModel):
    def __init__(self, num_classes, model_name_to_download, pretrained=True, **kwargs):
        feature_embedding_modules = [
            ResNet,
            AvgPoolFlexibleDimension,
            AvgPoolFlexibleDimension,
        ]
        feature_embeddings_args = [
            dict(model_name_to_download=model_name_to_download, pretrained=pretrained),
            dict(dim=2),
            dict(dim=2),
        ]
        super(EasyPeasyResNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class EasyPeasyConvNet(ClassificationModel):
    def __init__(self, num_classes, kernel_size, filter_list, stride, padding):
        feature_embedding_modules = [Conv2dEmbedding]
        feature_embeddings_args = [
            dict(
                kernel_size=kernel_size,
                layer_filter_list=filter_list,
                stride=stride,
                padding=padding,
            )
        ]
        super(EasyPeasyConvNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class EasyPeasyConvRelationalNet(ClassificationModel):
    def __init__(
        self,
        num_classes,
        kernel_size,
        filter_list,
        stride,
        padding,
        relational_num_filters,
        relational_num_layers,
        relational_num_outputs,
    ):
        feature_embedding_modules = [Conv2dEmbedding, BatchRelationalModule]
        feature_embeddings_args = [
            dict(
                kernel_size=kernel_size,
                layer_filter_list=filter_list,
                stride=stride,
                padding=padding,
            ),
            dict(
                num_filters=relational_num_filters,
                num_layers=relational_num_layers,
                num_outputs=relational_num_outputs,
                num_post_processing_filters=relational_num_filters,
                num_post_processing_layers=relational_num_layers,
                bias=True,
                avg_pool_input_shape=None,
            ),
        ]
        super(EasyPeasyConvRelationalNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )
