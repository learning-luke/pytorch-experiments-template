from __future__ import print_function

import os
import pathlib
from collections import OrderedDict

import torch
import torch.nn as nn
from rich import print

from models.auto_builder_models import ClassificationModel
from models.clip_models.model import Transformer, LayerNorm, model_to_download_url_dict
from utils.storage import download_file
import torch.nn.functional as F
import torch
import torch.nn as nn


class VisualTransformer(nn.Module):
    def __init__(
        self,
        grid_patch_size: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        stem_conv_bias: False,
        model_name_to_download: None,
        pretrained: False,
    ):
        super().__init__()
        self.grid_patch_size = grid_patch_size
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.model_name_to_download = model_name_to_download
        self.pretrained = pretrained
        self.stem_conv_bias = stem_conv_bias
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)
        out = dummy_x

        if self.pretrained and out.shape[2] != 224:
            out = F.interpolate(out, size=(224, 224))

        self.conv1 = nn.Conv2d(
            in_channels=out.shape[1],
            out_channels=self.transformer_num_filters,
            kernel_size=self.grid_patch_size,
            stride=self.grid_patch_size,
            bias=self.stem_conv_bias,
        )

        scale = self.transformer_num_filters ** -0.5

        self.class_embedding = nn.Parameter(
            scale * torch.randn(self.transformer_num_filters)
        )

        self.num_patches = (input_shape[2] // self.grid_patch_size) ** 2 + 1

        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                self.num_patches,
                self.transformer_num_filters,
            )
        )

        self.ln_pre = LayerNorm(self.transformer_num_filters)

        out = self.conv1(out)  # shape = [*, width, grid, grid]
        out = out.reshape(
            out.shape[0], out.shape[1], -1
        )  # shape = [*, width, grid ** 2]
        out = out.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        out = torch.cat(
            [
                self.class_embedding.to(out.dtype)
                + torch.zeros(
                    out.shape[0], 1, out.shape[-1], dtype=out.dtype, device=out.device
                ),
                out,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        out = out + self.positional_embedding.to(out.dtype)

        self.transformer = Transformer(
            width=self.transformer_num_filters,
            layers=self.transformer_num_layers,
            heads=self.transformer_num_heads,
        )

        self.ln_post = LayerNorm(self.transformer_num_filters)

        out = self.ln_pre(out)

        out = out.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(out)
        out = out.permute(1, 0, 2)  # LND -> NLD

        b, s, f = out.shape

        out = self.ln_post(out.view(b * s, f))

        out = out.view(b, s, f)

        self.is_built = True

        print("Built", self.__class__.__name__, "with output shape", out.shape)

        if self.pretrained:
            self._load_pretrained_model()

    def _load_pretrained_model(self):
        print("Loading pretrained model weights from", self.model_name_to_download)
        weights_directory = os.path.join(
            os.getcwd(), "models", "clip_models", "pretrained_weights"
        )
        weights_directory = pathlib.Path(weights_directory)
        weights_directory.mkdir(parents=True, exist_ok=True)

        model_weights_filepath = os.path.join(
            weights_directory, f"{self.model_name_to_download}.pt"
        )

        if not os.path.exists(model_weights_filepath):
            print(model_weights_filepath)
            target_url = model_to_download_url_dict[self.model_name_to_download]
            download_file(url=target_url, filename=model_weights_filepath, verbose=True)
        test_dict = {
                key: value
                for key, value in torch.load(
                    model_weights_filepath
                ).visual.named_parameters()
            }
        for key, value in test_dict.items():
            print(key, value.shape)
        named_parameters = OrderedDict(
            {
                key: value
                for key, value in torch.load(
                    model_weights_filepath
                ).visual.named_parameters()
            }
        )
        self.load_state_dict(state_dict=named_parameters, strict=False)

    def forward(self, x: torch.Tensor):

        if self.pretrained and x.shape[2] != 224:
            x = F.interpolate(x, size=(224, 224))


        if not self.is_built:
            self.build(input_shape=x.shape)

        out = x
        out = self.conv1(out)  # shape = [*, width, grid, grid]
        out = out.reshape(
            out.shape[0], out.shape[1], -1
        )  # shape = [*, width, grid ** 2]
        out = out.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        out = torch.cat(
            [
                self.class_embedding.to(out.dtype)
                + torch.zeros(
                    out.shape[0], 1, out.shape[-1], dtype=out.dtype, device=out.device
                ),
                out,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        out = out + self.positional_embedding.to(out.dtype)
        out = self.ln_pre(out)

        out = out.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(out)
        out = out.permute(1, 0, 2)  # LND -> NLD

        b, s, f = out.shape

        out = self.ln_post(out.view(b * s, f))

        out = out.view(b, s, f)

        return out


class EasyPeasyViTFlatten(ClassificationModel):
    def __init__(
        self,
        num_classes,
        grid_patch_size: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        model_name_to_download: str,
        stem_conv_bias=False,
        pretrained=True,
        **kwargs,
    ):
        feature_embedding_modules = [VisualTransformer]
        feature_embeddings_args = [
            dict(
                grid_patch_size=grid_patch_size,
                transformer_num_filters=transformer_num_filters,
                transformer_num_layers=transformer_num_layers,
                transformer_num_heads=transformer_num_heads,
                model_name_to_download=model_name_to_download,
                pretrained=pretrained,
                stem_conv_bias=stem_conv_bias,
            ),
        ]
        super(EasyPeasyViTFlatten, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class ChooseSpecificTimeStepFromVector(nn.Module):
    def __init__(self, time_step_to_choose):
        super(ChooseSpecificTimeStepFromVector, self).__init__()
        self.time_step_to_choose = time_step_to_choose

    def forward(self, x):
        return x[:, self.time_step_to_choose, :]


class EasyPeasyViTLastTimeStep(ClassificationModel):
    def __init__(
        self,
        num_classes,
        grid_patch_size: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        model_name_to_download: str,
        stem_conv_bias=False,
        pretrained=True,
        **kwargs,
    ):
        feature_embedding_modules = [
            VisualTransformer,
            ChooseSpecificTimeStepFromVector,
        ]
        feature_embeddings_args = [
            dict(
                grid_patch_size=grid_patch_size,
                transformer_num_filters=transformer_num_filters,
                transformer_num_layers=transformer_num_layers,
                transformer_num_heads=transformer_num_heads,
                model_name_to_download=model_name_to_download,
                pretrained=pretrained,
                stem_conv_bias=stem_conv_bias,
            ),
            dict(time_step_to_choose=0),
        ]
        super(EasyPeasyViTLastTimeStep, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )
