'''
Adapted from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py,
and my own work
'''
import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils.storage import build_experiment_folder, save_statistics, save_checkpoint, restore_model, \
    get_start_epoch, get_best_epoch, save_activations, save_image_batch, print_network_stats
from models.model_selector import ModelSelector
from utils.datasets import load_dataset
from utils.administration import parse_args
import random
from torchvision.utils import save_image
from utils.torchsummary import summary
args = parse_args()

######################################################################################################### Seeding
# Seeding can be annoying in pytorch at the moment. Based on my experience, the below means of seeding
# allows for deterministic experimentation.
torch.manual_seed(args.seed)
np.random.seed(args.seed)  # set seed
random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
if device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

######################################################################################################### Data
trainloader, testloader, in_shape = load_dataset(args)
n_train_batches = len(trainloader)
n_train_images = len(trainloader.dataset)
n_test_batches = len(testloader)
n_test_images = len(testloader.dataset)

print("Data loaded successfully ")
print("Training --> {} images and {} batches".format(n_train_images, n_train_batches))
print("Testing --> {} images and {} batches".format(n_test_images, n_test_batches))

######################################################################################################### Admin
# Build folders
saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(args)

# Always save a snapshot of the current state of the code. I've found this helps immensely if you find that one of your many experiments was actually quite good but you forgot what you did
import glob
import tarfile
snapshot_filename = '{}/snapshot.tar.gz'.format(saved_models_filepath)
filetypes_to_include = ['.py']
all_files = []
for filetype in filetypes_to_include:
    all_files += glob.glob('**/*.py', recursive=True)
with tarfile.open(snapshot_filename, "w:gz") as tar:
    for file in all_files:
        tar.add(file)

# For resuming the model training, find out from where
start_epoch, latest_loadpath = get_start_epoch(args)
args.latest_loadpath = latest_loadpath
best_epoch, best_test_acc = get_best_epoch(args)
if best_epoch >= 0:
    print('Best evaluation acc so far at {} epochs: {:0.2f}'.format(best_epoch, best_test_acc))

if not args.resume:
    # These are the currently tracked stats. I'm sure there are cleaner ways of doing this though.
    save_statistics(logs_filepath, "result_summary_statistics",
                    ["epoch",
                     "train_loss",
                     "test_loss",
                     "train_loss_c",
                     "test_loss_c",
                     "train_acc",
                     "test_acc",
                     ],
                    create=True)

######################################################################################################### Model
num_classes = 10 if args.dataset != 'Cifar-100' else 100
net = ModelSelector(in_shape=in_shape,
                    num_classes=num_classes).select(args.model, args)
print_network_stats(net)

print('Network summary:')
summary(net, (in_shape[2], in_shape[0], in_shape[1]), args.batch_size)

net = net.to(device)

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
    'optimizer':optimizer,
}

if args.resume:
    restore_model(restore_fields, args)
######################################################################################################### Training


def get_losses(inputs, targets):
    """
    It tends to be much easier to calculate losses, particularly considering there may be many of these, in a function.
    :param inputs: Input images, X
    :param targets: Input targets, y
    :return: Losses, logits, activations, and targets (in case of a change of targets)
    """

    logits, activations = net(inputs)
    loss = criterion(logits, targets)
    return (loss, ), logits, activations, targets

def run_epoch(epoch, train=True):
    global net
    if train:
        net.train()
    else:
        net.eval()
    total_loss = 0
    total_loss_c = 0
    correct = 0
    total = 0
    batches = n_train_batches if train else n_test_batches
    identifier = 'train' if train else 'test'
    with tqdm.tqdm(initial=0, total=batches) as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader if train else testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            losses, logits, activations, targets = get_losses(inputs, targets)

            loss_c = losses[0]
            loss = loss_c

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_loss_c += loss_c.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            iter_out = '{}, {}: {}; Loss: {:0.4f}, Loss_c: {:0.4f}, Acc: {:0.4f}'.format(
                args.exp_name,
                identifier,
                batch_idx,
                total_loss / (batch_idx + 1),
                total_loss_c / (batch_idx + 1),
                100. * correct / total,
            )

            pbar.set_description(iter_out)
            pbar.update()

            if args.save_images and batch_idx==0:
                # Would save any images here under '{}/{}/{}_stuff.png'.format(images_filepath, identifier, epoch)
                pass

    return total_loss / batches, total_loss_c / batches, correct / total


if __name__ == "__main__":
    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        for epoch in range(start_epoch, args.max_epochs):
            scheduler.step(epoch=epoch)

            train_loss, train_loss_c, train_acc = run_epoch(epoch, train=True)
            test_loss, test_loss_c, test_acc = run_epoch(epoch, train=False)

            save_statistics(logs_filepath, "result_summary_statistics",
                            [epoch,
                             train_loss,
                             test_loss,
                             train_loss_c,
                             test_loss_c,
                             train_acc,
                             test_acc])

            ############################################################################################## Saving models
            if args.save:
                state = {
                    'epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                epoch_pbar.set_description('Saving at {}/{}_checkpoint.pth.tar'.format(saved_models_filepath, epoch))
                filename = '{}_checkpoint.pth.tar'.format(epoch)
                previous_save = '{}/{}_checkpoint.pth.tar'.format(saved_models_filepath, epoch - 1)
                if os.path.isfile(previous_save):
                    os.remove(previous_save)
                save_checkpoint(state=state,
                                directory=saved_models_filepath,
                                filename=filename,
                                is_best=False)
            ############################################################################################################

            epoch_pbar.set_description('')
            epoch_pbar.update(1)

