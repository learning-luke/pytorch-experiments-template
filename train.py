import argparse
import json
import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils.storage import build_experiment_folder, save_checkpoint, restore_model
from utils.data_loaders import load_dataset
import random
import glob
import tarfile
from models.wresnet import WideResNet
from utils.torchsummary import summary
from utils.metric_tracking import MetricTracker, compute_accuracy

def parse_args(verbose=True):
    """
    Argument parser
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument('-data', '--dataset', type=str, default='cifar10')
    parser.add_argument('-loc', '--data_loc', type=str, default='../data/Cifar-10')
    parser.add_argument('-batch', '--batch_size', type=int, default=20)
    parser.add_argument('-numw', '--num_workers', type=int, default=1)
    parser.add_argument('-tbatch', '--test_batch_size', type=int, default=100)
    parser.add_argument('-x', '--max_epochs', type=int, default=200)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-resume', '--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('-dist', '--distributed', dest='distributed', action='store_true')
    parser.set_defaults(distributed=False)
    parser.add_argument('-test', '--test', dest='test', action='store_true')
    parser.set_defaults(distributed=False)

    # logging
    parser.add_argument('-en', '--exp_name', type=str, default='dev')
    parser.add_argument('-o', '--logs_path', type=str, default='log')
    parser.add_argument('-save', '--save', dest='save', action='store_true')
    parser.add_argument('-nosave', '--nosave', dest='save', action='store_false')
    parser.set_defaults(save=True)

    # model
    parser.add_argument('-model', '--model', type=str, default='wresnet')
    parser.add_argument('-dep', '--resdepth', type=int, default=28)
    parser.add_argument('-wf', '--widen_factor', type=int, default=10)
    parser.add_argument('-dropout', '--dropout_rate', type=float, default=0.3)

    # optimization
    parser.add_argument('-l', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-sched', '--scheduler', type=str, default='MultiStep', help='Scheduler for learning rate annealing: CosineAnnealing | MultiStep')
    parser.add_argument('-mile', '--milestones', type=int, nargs='+', default=[60, 120, 160], help='Multi step scheduler annealing milestones')
    parser.add_argument('-optim', '--optim', type=str, default='SGD', help='Optimizer?')

    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('-mom', '--momentum', type=float, default=0.9)

    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    return args

args = parse_args()

######################################################################################################### Admin
saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(experiment_name=args.exp_name, log_path=args.logs_path)

######################################################################################################### Data

train_loader, val_loader, train_set, val_set, data_shape = load_dataset(args.dataset, args.data_loc, batch_size=args.batch_size, test_batch_size=args.test_batch_size, num_workers=args.num_workers, download=False, test=args.test)

######################################################################################################### Determinism
# Seeding can be annoying in pytorch at the moment. Based on my experience, the below means of seeding
# allows for deterministic experimentation.
torch.manual_seed(args.seed)
np.random.seed(args.seed)  # set seed
random.seed(args.seed)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
args.device = device
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

# Always save a snapshot of the current state of the code. I've found this helps immensely if you find that one of your
# many experiments was actually quite good but you forgot what you did

snapshot_filename = '{}/snapshot.tar.gz'.format(saved_models_filepath)
filetypes_to_include = ['.py']
all_files = []
for filetype in filetypes_to_include:
    all_files += glob.glob('**/*.py', recursive=True)
with tarfile.open(snapshot_filename, "w:gz") as tar:
    for file in all_files:
        tar.add(file)

######################################################################################################### Model

num_classes = 100 if args.dataset.lower() == 'cifar100' else 10
net = WideResNet(depth=args.resdepth, num_classes=num_classes, widen_factor=10, dropRate=args.dropout_rate)
if args.distributed:
    net = nn.DataParallel(net)
net = net.to(device)
summary(net, in_shape, batch_size=args.batch_size)

######################################################################################################### Optimisation

params = net.parameters()
criterion = nn.CrossEntropyLoss()

if args.optim.lower() == 'sgd':
    optimizer = optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adam(params, lr=args.learning_rate, amsgrad=True, weight_decay=args.weight_decay)

if args.scheduler == 'CosineAnnealing':
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epochs, eta_min=0)
else:
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

######################################################################################################### Restoring

restore_fields = {
    'net': net,
    'optimizer': optimizer,
    'scheduler':scheduler,
}

start_epoch = 0
if args.resume:
    resume_epoch = restore_model(restore_fields, path=saved_models_filepath)
    if resume_epoch == -1:
        print("Failed to load from {}/ckpt.pth.tar".format(saved_models_filepath))
    else:
        start_epoch = resume_epoch+1

######################################################################################################### Metric

metrics_to_track={'cross_entropy': lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(), 'accuracy':compute_accuracy}
metric_tracker_train = MetricTracker(metrics_to_track=metrics_to_track, load=True if start_epoch>0 else False, path='{}/metrics_train.pt'.format(logs_filepath))
metric_tracker_test = MetricTracker(metrics_to_track=metrics_to_track, load=True if start_epoch>0 else False, path='{}/metrics_test.pt'.format(logs_filepath))

######################################################################################################### Training

def train_iter(net, x, y, iteration, epoch, set_name):
    global metric_tracker_train

    inputs, targets = x.to(device), y.to(device)

    net = net.train()

    logits, activations = net(inputs)

    loss = criterion(input=logits, target=targets)
    metric_tracker_train.push(epoch, iteration, logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_string = '{}, {}: {}; {}'.format(
        args.exp_name,
        set_name,
        iteration,
        ''.join([('{}: {:0.4f}; '.format(key, value[-1]) if (key != 'epochs' and key != 'iterations') else '')
                   for key, value in metric_tracker_train.metrics.items()]),
    )

    return log_string


def eval_iter(net, x, y, iteration, epoch, set_name):
    global metric_tracker_test
    x, targets = x.to(device), y.to(device)

    net = net.eval()

    logits, activations = net(x)

    metric_tracker_test.push(epoch, iteration, logits, targets)


    log_string = '{}, {}: {}; {}'.format(
        args.exp_name,
        set_name,
        iteration,
        ''.join([('{}: {:0.4f}; '.format(key, value[-1]) if (key != 'epochs' and key != 'iterations') else '')
                   for key, value in metric_tracker_test.metrics.items()]),
    )

    return log_string

train_iterations = 0
def run_epoch(epoch, net, train=True):
    global train_iterations
    identifier = 'train' if train else 'test'
    loader = train_loader if train else val_loader
    with tqdm.tqdm(initial=0, total=len(loader)) as pbar:

        for batch_idx, (inputs, targets) in enumerate(loader):

            if train:

                log_string = train_iter(net=net, x=inputs, y=targets,
                                        iteration=train_iterations, epoch=epoch,
                                        set_name=identifier)
                train_iterations += 1
            else:
                log_string = eval_iter(net=net, x=inputs, y=targets,
                                       iteration=train_iterations, epoch=epoch,
                                       set_name=identifier)

            pbar.set_description(log_string)
            pbar.update()





if __name__ == "__main__":
    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        for epoch in range(start_epoch, args.max_epochs):

            run_epoch(epoch, net=net, train=True)
            run_epoch(epoch, net=net, train=False)
            scheduler.step()



            metric_tracker_train.plot(path='{}/train/metrics.png'.format(images_filepath))
            metric_tracker_test.plot(path='{}/test/metrics.png'.format(images_filepath))
            metric_tracker_train.save()
            metric_tracker_test.save()

################################################################################ Saving models
            if args.save:
                state = {
                    'args':args,
                    'epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),
                }
                epoch_pbar.set_description('Saving at {}/ckpt.pth.tar'.format(saved_models_filepath))
                filename = 'ckpt.pth.tar'.format(epoch)
                previous_save = '{}/ckpt.pth.tar'.format(saved_models_filepath, epoch - 1)
                if os.path.isfile(previous_save):
                    os.remove(previous_save)
                save_checkpoint(state=state,
                                directory=saved_models_filepath,
                                filename=filename,
                                is_best=False)
            ############################################################################################################

            epoch_pbar.set_description('')
            epoch_pbar.update(1)
