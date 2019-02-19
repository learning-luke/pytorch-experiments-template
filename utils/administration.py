import argparse
import json


def parse_args():
    """
    Argument parser
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-data', '--dataset', type=str, default='Cifar-10',
                        help='Which dataset to use')
    parser.add_argument('-norm', '--dataset_norm_type', type=str, default='standardize',
                        help='How to normalize data? Standardize | one')
    parser.add_argument('-batch', '--batch_size', type=int, default=128,
                        help='Batch Size')
    parser.add_argument('-tbatch', '--test_batch_size', type=int, default=100,
                        help='Test Batch Size')
    parser.add_argument('-x', '--max_epochs', type=int, default=200,
                        help='How many args.max_epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed to use')
    parser.add_argument('-aug', '--data_aug', type=str, nargs='+',
                        default=['NONE'],#['random_h_flip', 'random_crop'],
                        help='Data augmentation?')
    # logging
    parser.add_argument('-en', '--exp_name', type=str, default='tester',
                        help='Experiment name for the model to be assessed')
    parser.add_argument('-o', '--logs_path', type=str, default='log',
                        help='Directory to save log files, check points, and tensorboard.')
    parser.add_argument('-resume', '--resume', type=int, default=0,
                        help='Resume training?')
    parser.add_argument('-save', '--save', type=int, default=0,
                        help='Save checkpoint files?')
    parser.add_argument('-imgs', '--save_images', type=int, default=1,
                        help='Save images?')
    # model
    parser.add_argument('-model', '--model', type=str, default='densenet',
                        help='resnet | preact_resnet | densenet | wresnet')
    # resnet models
    parser.add_argument('-dep', '--resdepth', type=int, default=121,
                        help='ResNet default depth')
    parser.add_argument('-wf', '--widen_factor', type=int, default=2,
                        help='Wide resnet widen factor')

    # simple cnn model
    parser.add_argument('-act', '--activation', type=str, default='leaky_relu',
                        help='Activation function')
    parser.add_argument('-fil', '--filters', type=int, nargs='+', default=[64, 128, 256, 512],
                        help='Filters')
    parser.add_argument('-str', '--strides', type=int, nargs='+', default=[2, 2, 2, 2],
                        help='Strides')
    parser.add_argument('-ker', '--kernel_sizes', type=int, nargs='+', default=[3, 3, 3, 3],
                        help='Strides')
    parser.add_argument('-lin', '--linear_widths', type=int, nargs='+', default=[256, 128],
                        help='Additional linear layer widths. If empty, cnn goes from conv outs to single linear layer')
    parser.add_argument('-bn', '--use_batch_norm', type=int, default=0,
                        help='Use batch norm for simple CNN?')
    # optimization
    parser.add_argument('-l', '--learning_rate', type=float, default=0.1,
                        help='Base learning rate')
    parser.add_argument('-sched', '--scheduler', type=str, default='MultiStep',
                        help='Scheduler for learning rate annealing: CosineAnnealing | MultiStep')
    parser.add_argument('-mile', '--milestones', type=int, nargs='+', default=[60, 120, 160],
                        help='Multi step scheduler annealing milestones')
    parser.add_argument('-optim', '--optim', type=str, default='SGD',
                        help='Optimizer?')
    # simple regularisation
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4,
                        help='Weight decay value')
    parser.add_argument('-mom', '--momentum', type=float, default=0.9,
                        help='Weight decay value')
    # label smoothing
    parser.add_argument('-lsr', '--label_smoothing_regularisation', type=int, default=0,
                        help='Label smoothing regularisation')
    parser.add_argument('-lsr_eps', '--label_smoothing_regularisation_eps', type=float, default=0.1,
                        help='Label smoothing regularisation, epsilon')
    # dropblock
    parser.add_argument('-dp', '--dropblock_probability', type=float, default=0.,
                        help='Maximum dropblock probability')
    # manifold mixup
    parser.add_argument('-mmix', '--use_manifold_mixup', type=int, default=0,
                        help='Use manifold mixup')
    parser.add_argument('-mmix_alpha', '--manifold_mixup_alpha', type=float, default=0.2,
                        help='Manifold mixup alpha')
    parser.add_argument('-mmix_eligible', '--manifold_mixup_eligible', type=int, default=2,
                        help='Eligible layers for manifold mixup')

    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    return args
