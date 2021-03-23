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
        input_shape=(2, 32, 32, 3),
        num_classes=10,
    ):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def select(self, model_type, args):
        """
        Selector utility to create models from model directory
        :param model_type: which model to select. Currently choices are: (cnn | resnet | preact_resnet | densenet | wresnet)
        :return: neural network to be trained
        """
        if model_type == "cnn":
            model_type = SimpleModel(
                in_shape=self.input_shape,
                activation=args.activation,
                num_classes=self.num_classes,
                filters=args.filters,
                strides=args.strides,
                kernel_sizes=args.kernel_sizes,
                linear_widths=args.linear_widths,
                use_batch_norm=args.use_batch_norm,
            )
        else:
            if model_type == "resnet":
                assert args.depth in [
                    18,
                    34,
                    50,
                    101,
                    152,
                ], "Non-standard and unsupported resnet depth ({})".format(
                    args.depth
                )
                if args.depth == 18:
                    model_type = ResNet18(self.num_classes)
                elif args.depth == 34:
                    model_type = ResNet34(self.num_classes)
                elif args.depth == 50:
                    model_type = ResNet50(self.num_classes)
                elif args.depth == 101:
                    model_type = ResNet101(self.num_classes)
                else:
                    model_type = ResNet152()
            elif model_type == "densenet":
                assert args.depth in [
                    121,
                    161,
                    169,
                    201,
                ], "Non-standard and unsupported densenet depth ({})".format(
                    args.depth
                )
                if args.depth == 121:
                    model_type = DenseNet121(
                        growth_rate=12, num_classes=self.num_classes
                    )  # NB NOTE: growth rate controls cifar implementation
                elif args.depth == 161:
                    model_type = DenseNet161(growth_rate=12, num_classes=self.num_classes)
                elif args.depth == 169:
                    model_type = DenseNet169(growth_rate=12, num_classes=self.num_classes)
                else:
                    model_type = DenseNet201(growth_rate=12, num_classes=self.num_classes)
            elif model_type == "preact_resnet":
                assert args.depth in [
                    18,
                    34,
                    50,
                    101,
                    152,
                ], "Non-standard and unsupported preact resnet depth ({})".format(
                    args.depth
                )
                if args.depth == 18:
                    model_type = PreActResNet18(self.num_classes)
                elif args.depth == 34:
                    model_type = PreActResNet34(self.num_classes)
                elif args.depth == 50:
                    model_type = PreActResNet50(self.num_classes)
                elif args.depth == 101:
                    model_type = PreActResNet101(self.num_classes)
                else:
                    model_type = PreActResNet152()
            elif model_type == "wresnet":
                assert (
                    args.depth - 4
                ) % 6 == 0, "Wideresnet depth of {} not supported, must fulfill: (depth - 4) % 6 = 0".format(
                    args.depth
                )
                model_type = WideResNet(
                    depth=args.depth,
                    num_classes=self.num_classes,
                    widen_factor=args.widen_factor,
                )
            else:
                raise NotImplementedError("Model {} not supported".format(model_type))
        return model_type
