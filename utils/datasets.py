from torchvision import datasets, transforms
from torch.utils.data import Dataset, Sampler, SubsetRandomSampler
import torch
from utils.augmentors import Cutout
from utils.cinic_utils import enlarge_cinic_10, download_cinic
import numpy as np


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
       indices: Indices to sample from
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class CustomDataset(Dataset):
    def __init__(self,
                 which_set='Cifar-10',
                 root=None,
                 train=True,
                 download=True,
                 return_idxs=False,
                 num_classes=10,
                 aug=('random_order', 'random_h_flip', 'random_crop', 'random_rot_10', 'random_scale_0.9_1.1', 'random_shear_5', 'cutout'),
                 cut_n_holes=1,
                 cut_length=16,
                 dataset_norm_type='standardize'
                 ):

        image_length = 28 if 'MNIST' in which_set else 32
        self.norm_means, self.norm_stds = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        if dataset_norm_type == 'zeroone':
            self.norm_means, self.norm_stds = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        if 'MNIST' in which_set:
            self.norm_means, self.norm_stds = (0.1307,), (0.3081,)
            if dataset_norm_type == 'zeroone':
                self.norm_means, self.norm_stds = (0.5,), (0.5,)
        normalizer = transforms.Normalize(self.norm_means, self.norm_stds)

        transforms_list = []
        for augment in aug:
            # First do the things that don't change where the image is in the box
            if augment == 'random_h_flip':
                transforms_list.append(transforms.RandomHorizontalFlip())
            if augment == 'random_v_flip':
                transforms_list.append(transforms.RandomVerticalFlip())
            # Then mess with brightness etc.
            if augment == 'color_jitter':
                transforms_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.0))
            # Now do some sheering/cropping/rotation that changes where the images is
            if augment == 'affine':
                rot_degrees = 0
                scale_low = None
                scale_high = None
                shear_degrees = None
                for augment_inner in aug:
                    if 'random_rot' in augment_inner:
                        rot_degrees = int(augment_inner.split('_')[-1])
                    if 'random_scale' in augment_inner:
                        scale_low = float(augment_inner.split('_')[-2])
                        scale_high = float(augment_inner.split('_')[-1])
                    if 'random_shear' in augment_inner:
                        shear_degrees = int(augment_inner.split('_')[-1])

                transforms_list.append(transforms.RandomAffine(degrees=rot_degrees,
                                                               scale=None if (scale_low is None) or (scale_high is None) else (scale_low, scale_high),
                                                               shear=shear_degrees))
            if augment == 'random_crop':
                transforms_list.append(transforms.RandomCrop(size=[image_length, image_length], padding=4))

        transform = transforms.Compose(transforms_list) if 'random_order' not in aug else \
            transforms.Compose([transforms. RandomOrder(transforms=transforms_list)])

        transform.transforms.append(transforms.ToTensor())
        transform.transforms.append(normalizer)

        for augment in aug:
            # Finally do things that are related to regularisation
            if augment == 'cutout':
                transform.transforms.append(Cutout(n_holes=cut_n_holes, length=cut_length))

        if which_set == 'MNIST':
            self.dataset = datasets.MNIST(root='../data/MNIST' if root is None else root,
                                          train=train,
                                          download=download,
                                          transform=transform)

        elif which_set == 'Fashion-MNIST':
            self.dataset = datasets.FashionMNIST(root='../data/Fashion-MNIST' if root is None else root,
                                                 train=train,
                                                 download=download,
                                                 transform=transform)

        elif which_set == 'Cifar-100':
            self.dataset = datasets.CIFAR100(root='../data/Cifar-100' if root is None else root,
                                             train=train, download=download,
                                             transform=transform)
        elif 'Cinic-10' in which_set:
            root_to_cinic = '../data/Cinic-10' if root is None else root
            if download:
                download_cinic(root_to_cinic.replace('-enlarged',''))
                if '-enlarged' in which_set:
                    enlarge_cinic_10(root_to_cinic.replace('-enlarged',''))
            self.dataset = datasets.ImageFolder(root=('../data/Cinic-10' if root is None else root) + ('/train' if train else '/test'),
                                                transform=transform)
        else:
            self.dataset = datasets.CIFAR10(root='../data/Cifar-10' if root is None else root,
                                            train=train, download=download,
                                            transform=transform)

        self.return_idxs = return_idxs
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __getitem__(self, index):
        data, target = self.dataset[index]

        if not self.return_idxs:
            return data, target
        else:
            return data, target, index

    def __len__(self):
        return len(self.dataset)


def load_dataset(args):
    print('==> Preparing data..')
    assert args.dataset in ['Cifar-10', 'Cifar-100', 'Cinic-10', 'Cinic-10-enlarged', 'Fashion-MNIST', 'MNIST'], \
        "dataset {} not supported".format(args.dataset)
    in_shape = (28, 28, 1) if 'MNIST' in args.dataset else (32, 32, 3)  # Woohoo, great variety here.

    data_dir = args.root
    root = None if args.root.lower() == 'default' else '{}/{}'.format(data_dir, args.dataset)

    num_classes = 10
    if args.dataset == 'Cifar-100':
        num_classes = 100
    elif args.dataset == 'Cifar-100-20':
        num_classes = 20

    train_dataset = CustomDataset(which_set=args.dataset,
                                  root=root,
                                  train=True,
                                  download=True,
                                  aug=args.data_aug,
                                  dataset_norm_type=args.dataset_norm_type)

    train_idxs = np.arange(len(train_dataset))
    validation_p = 0.1
    num = int(validation_p * len(train_dataset) / num_classes)
    if args.test == 0:
        validation_idxs = np.array([])
        try:
            labels = train_dataset.dataset.train_labels
        except:
            labels = train_dataset.dataset.targets
        for ci in range(num_classes):
            idxs_this_ci = np.random.choice((np.array(labels) == ci).nonzero()[0], num, replace=False)
            validation_idxs = np.concatenate((validation_idxs, idxs_this_ci))
        train_idxs = np.setdiff1d(train_idxs, validation_idxs)

        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  sampler=SubsetRandomSampler(train_idxs),
                                                  num_workers=4,
                                                  pin_memory=True)

        testloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=args.test_batch_size,
                                                 sampler=SubsetSequentialSampler(list(validation_idxs.astype(int))),
                                                 pin_memory=True, )
    else:
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=8,
                                                  pin_memory=True,
                                                  drop_last=True,
                                                  )
        test_dataset = CustomDataset(which_set=args.dataset,
                                     root=root,
                                     train=False,
                                     download=True,
                                     aug=[],
                                     dataset_norm_type=args.dataset_norm_type, )
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.test_batch_size, shuffle=False,
                                                 pin_memory=True, )


    args.norm_means = train_dataset.norm_means
    args.norm_stds = train_dataset.norm_stds
    return trainloader, testloader, in_shape

