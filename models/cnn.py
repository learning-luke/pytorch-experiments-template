"""
Simple CNN implementation with flexible numbers of layers
"""
import torch.nn as nn


class CNNSimple(nn.Module):
    def __init__(
        self,
        in_shape=(28, 28, 1),
        filters=(32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        kernel_sizes=(3, 3, 3, 3),
        linear_widths=(256,),
        activation=nn.LeakyReLU(),
        num_classes=10,
        use_batch_norm=False,
    ):
        super(CNNSimple, self).__init__()
        self.activation = activation
        conv_list = []
        out_spatial_dims = in_shape[
            0
        ]  # To keep track of how big the spatial dims are, for later housekeeping

        for i, (filter, stride, kernel_size) in enumerate(
            zip(filters, strides, kernel_sizes)
        ):
            channels_in = in_shape[2] if i == 0 else filters[i - 1]
            channels_out = filter
            if use_batch_norm:
                conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            channels_in,
                            channels_out,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=1,
                            bias=False,
                        ),
                        activation,
                        nn.BatchNorm2d(channels_out),
                    )
                )
            else:
                conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            channels_in,
                            channels_out,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=1,
                            bias=True,
                        ),
                        activation,
                    )
                )
            out_spatial_dims //= stride

        self.conv_layers = nn.Sequential(*conv_list)
        # Now do the linear layers
        linear_widths = (
            [out_spatial_dims ** 2 * filters[-1]] + linear_widths + [num_classes]
        )
        linear_list = []
        for i in range(
            len(linear_widths) - 2
        ):  # Only go up to the second last width (no act on final)
            linear_list.append(
                nn.Sequential(
                    nn.Linear(linear_widths[i], linear_widths[i + 1]), activation
                )
            )
        linear_list.append(nn.Linear(linear_widths[-2], linear_widths[-1]))
        self.linear_layers = nn.Sequential(*linear_list)

    def forward(self, x):
        convolutions_out = self.conv_layers(x)
        logits = self.linear_layers(convolutions_out.view(convolutions_out.size(0), -1))
        return logits, (convolutions_out,)


def SimpleModel(
    in_shape=(28, 28, 1),
    activation="relu",
    filters=(32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_sizes=(3, 3, 3, 3),
    linear_widths=(256,),
    num_classes=10,
    use_batch_norm=False,
):
    if activation.lower() == "leaky_relu":
        activation = nn.LeakyReLU()
    elif activation.lower() == "prelu":
        activation = nn.PReLU()
    elif activation.lower() == "sigmoid":
        activation = nn.Sigmoid()
    else:
        activation = nn.ReLU()
    return CNNSimple(
        in_shape=in_shape,
        activation=activation,
        filters=filters,
        strides=strides,
        kernel_sizes=kernel_sizes,
        linear_widths=linear_widths,
        num_classes=num_classes,
        use_batch_norm=use_batch_norm,
    )
