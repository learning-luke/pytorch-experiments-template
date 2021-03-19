from models.cnn import SimpleModel
from models.preact_resnet import (
    PreActResNet18,
    PreActResNet34,
    PreActResNet50,
    PreActResNet101,
    PreActResNet152,
)
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.wresnet import WideResNet


class ModelSelector:
    """
    Selector class, so that you can call upon multiple models
    """

    def __init__(
        self,
        in_shape=(32, 32, 3),
        num_classes=10,
    ):
        self.num_classes = num_classes
        self.in_shape = in_shape

    def select(self, model, args):
        """
        Selector utility to create models from model directory
        :param model: which model to select. Currently choices are: (cnn | resnet | preact_resnet | densenet | wresnet)
        :return: neural network to be trained
        """
        if model == "cnn":
            net = SimpleModel(
                in_shape=self.in_shape,
                activation=args.activation,
                num_classes=self.num_classes,
                filters=args.filters,
                strides=args.strides,
                kernel_sizes=args.kernel_sizes,
                linear_widths=args.linear_widths,
                use_batch_norm=args.use_batch_norm,
            )
        else:
            assert (
                args.dataset != "MNIST" and args.dataset != "Fashion-MNIST"
            ), "Cannot use resnet or densenet for mnist style data"
            if model == "resnet":
                assert args.resdepth in [
                    18,
                    34,
                    50,
                    101,
                    152,
                ], "Non-standard and unsupported resnet depth ({})".format(
                    args.resdepth
                )
                if args.resdepth == 18:
                    net = ResNet18(self.num_classes)
                elif args.resdepth == 34:
                    net = ResNet34(self.num_classes)
                elif args.resdepth == 50:
                    net = ResNet50(self.num_classes)
                elif args.resdepth == 101:
                    net = ResNet101(self.num_classes)
                else:
                    net = ResNet152()
            elif model == "densenet":
                assert args.resdepth in [
                    121,
                    161,
                    169,
                    201,
                ], "Non-standard and unsupported densenet depth ({})".format(
                    args.resdepth
                )
                if args.resdepth == 121:
                    net = DenseNet121(
                        growth_rate=12, num_classes=self.num_classes
                    )  # NB NOTE: growth rate controls cifar implementation
                elif args.resdepth == 161:
                    net = DenseNet161(growth_rate=12, num_classes=self.num_classes)
                elif args.resdepth == 169:
                    net = DenseNet169(growth_rate=12, num_classes=self.num_classes)
                else:
                    net = DenseNet201(growth_rate=12, num_classes=self.num_classes)
            elif model == "preact_resnet":
                assert args.resdepth in [
                    18,
                    34,
                    50,
                    101,
                    152,
                ], "Non-standard and unsupported preact resnet depth ({})".format(
                    args.resdepth
                )
                if args.resdepth == 18:
                    net = PreActResNet18(self.num_classes)
                elif args.resdepth == 34:
                    net = PreActResNet34(self.num_classes)
                elif args.resdepth == 50:
                    net = PreActResNet50(self.num_classes)
                elif args.resdepth == 101:
                    net = PreActResNet101(self.num_classes)
                else:
                    net = PreActResNet152()
            elif model == "wresnet":
                assert (
                    args.resdepth - 4
                ) % 6 == 0, "Wideresnet depth of {} not supported, must fulfill: (depth - 4) % 6 = 0".format(
                    args.resdepth
                )
                net = WideResNet(
                    depth=args.resdepth,
                    num_classes=self.num_classes,
                    widen_factor=args.widen_factor,
                )
            else:
                raise NotImplementedError("Model {} not supported".format(model))
        return net
