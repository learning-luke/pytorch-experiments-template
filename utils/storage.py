"""
Storage associated utilities
"""
import numpy as np
import scipy.misc
import shutil
import torch
from collections import OrderedDict
import scipy
import json
import os


def isfloat(x):
    return isinstance(x, float)


def isint(x):
    return isinstance(x, int)


def save_metrics_dict_in_pt(log_dir, metrics_file_name, metrics_dict, overwrite):
    """
    Saves a metrics .pt file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param metrics_dict: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath, if False adds metrics to existing
    """
    if not metrics_file_name.endswith('.pt'):
        metrics_file_name = '{}.pt'.format(metrics_file_name)

    metrics_file_path = os.path.join(log_dir, metrics_file_name)

    if overwrite:
        if os.path.exists(metrics_file_path):
            os.remove(metrics_file_path)

    torch.save(metrics_dict, metrics_file_path)


def load_metrics_dict_from_pt(log_dir, metrics_file_name):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if not metrics_file_name.endswith('.pt'):
        metrics_file_name = '{}.pt'.format(metrics_file_name)

    metrics_file_path = os.path.join(log_dir, metrics_file_name)

    metrics_dict = torch.load(metrics_file_path)

    return metrics_dict

def save_image_batch(filename, images, clip=True, types=('features', 'gray'), norm_means=(0.5, 0.5, 0.5),
                     norm_stds=(0.5, 0.5, 0.5)):
    """
    Save utility for a batch of different kinds of images. Pretty useful.

    If you have multiple batches of images, use it like this
    save_image_batch('save.png', images=torch.stack((images1, images2)), types=('color', 'colour'))

    But if you have only one batch of images, use it like this:
    save_image_batch('save.png', images=images1.view((1,) + images1.shape), types=('colour',))

    :param filename: where to save
    :param images: input batch of images as Tensors of: [R, B, C, H, W], where
            R - Number of batches to save
            B - Batch size
            C - Channels - either 1 or 3 (if you want to save features, reshape these first)
            W - Width (spatial)
            H - Height (spatial)
    :param clip: clip between 0 and 1, sometimes necessary
    :param types: The input types for the different batches. Can be 'colour', 'gray', or 'features'
    :param norm_means: normalisation values used when creating the dataset, these are used to unnorm the data
    :param norm_stds: normalisation values used when creating the dataset, these are used to unnorm the data
    :return: Nothing, only save
    """

    import matplotlib.pyplot as plt
    colormap = plt.get_cmap('inferno')
    repeats = images.size(0)
    assert len(types) == repeats, 'must have the same number of types listed as repeats'
    n = images.size(1)
    width = int(np.round(np.sqrt(n)))
    height = int(np.ceil(n / width))

    img_w = images.size(3)
    img_h = images.size(4)
    img_c = images.size(2)
    buffer = 1 if img_w > 1 else 0

    actual_height = buffer + img_h * height + buffer * height
    output_img = np.zeros(shape=(buffer + img_w * width + buffer * width, actual_height * repeats, 3)) + 1
    if img_c == 1:
        output_img *= 0.5

    minimum = torch.min(images).item()
    maximum = torch.max(images).item() - minimum

    for repeat in range(repeats):
        type_repeat = types[repeat]

        for i, img in enumerate(images[repeat]):
            x, y = np.unravel_index(i, dims=(width, height))
            img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
            if type_repeat == 'colour':
                img[:, :, 0] = img[:, :, 0] * norm_stds[0] + norm_means[0]
                img[:, :, 1] = img[:, :, 1] * norm_stds[1] + norm_means[1]
                img[:, :, 2] = img[:, :, 2] * norm_stds[2] + norm_means[2]
                img = np.clip(img, 0, 1)
            elif type_repeat == 'gray':
                img = np.clip(img, 0, 1)
            elif type_repeat == 'features':
                img -= minimum
                img /= maximum
                img = colormap(img)
                img = img[:, :, 0][:, :, 0:3]
            output_img[buffer + x * (img_w + buffer):buffer + (x) * (img_w + buffer) + img_w,
            actual_height * repeat + buffer + y * (img_h + buffer): actual_height * repeat + buffer + y * (
                    img_h + buffer) + img_h, :] = img

    scipy.misc.toimage(np.squeeze(output_img)).save(filename)


def save_activations(filename, inputs, activations, num_images=20, buffer=2):
    """
    Simple activation visualisation for saving activaitons
    :param filename: where to save
    :param inputs: input images
    :param activations: activations from the network, always a greater number than num_images
    :param num_images: the number of activations to visualise
    :param buffer: the buffer size between each image
    :return: Nothing, only save
    """
    import matplotlib.pyplot as plt
    import scipy.misc
    import torch.nn.functional as F
    colormap = plt.get_cmap('viridis')

    cols = len(activations) + 1
    rows = num_images
    image_size = inputs.size(2)
    width = cols * image_size + buffer * (cols - 1)
    height = rows * image_size + buffer * (rows - 1)

    activations = [F.upsample(att, (image_size, image_size)) for att in activations]

    inputs_extended = torch.zeros((activations[0].size(0), *inputs.shape[1:]))
    inputs_extended[:inputs.shape[0], :, :, :] = inputs

    output_img = np.zeros((height, width, 3))
    for i, output in enumerate(zip(inputs_extended, *activations)):
        output = list(output)
        img = output[0]  # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        atts = output[1:]
        img = np.transpose(img, (1, 2, 0))
        img[:, :, 0] = img[:, :, 0] * 0.2023 + 0.4914
        img[:, :, 1] = img[:, :, 1] * 0.1994 + 0.4822
        img[:, :, 2] = img[:, :, 2] * 0.2010 + 0.4465
        output_img[i * (image_size + buffer):i * (image_size + buffer) + image_size, 0:image_size, :] = img
        for ai in range(len(atts)):
            minimum = torch.min(atts[ai]).item()
            maximum = torch.max(atts[ai]).item() - minimum
            atts[ai] = np.squeeze(atts[ai].detach().cpu().numpy())
            if maximum != minimum:
                atts[ai] -= minimum
                atts[ai] /= maximum
            atts[ai] = colormap(atts[ai])[:, :, 0:3]
            output_img[i * (image_size + buffer):i * (image_size + buffer) + image_size,
            (ai + 1) * (image_size + buffer):(ai + 1) * (image_size + buffer) + image_size, :] = atts[ai]
        if (i + 1) == num_images:
            break

    scipy.misc.imsave(filename, output_img)


def save_checkpoint(state, is_best, directory='', filename='checkpoint.pth.tar'):
    """
    Checkpoint saving utility, to ensure that the checkpoints are saved in the right place
    :param state: this is what gets saved.
    :param is_best: if this is the current best model, save a copy of it with a `best_` 
    :param directory: where to save
    :param filename: using this filename
    :return: nothing, just save things
    """
    save_path = '{}/{}'.format(directory, filename) if directory != '' else filename
    torch.save(state, save_path)

    if is_best:
        best_save_path = '{}/best_{}'.format(directory, filename) if directory != '' else 'best_{}'.format(filename)
        shutil.copyfile(save_path, best_save_path)


def restore_model(restore_fields, args):
    """
    Model restoration. This is built into the experiment framework and args.latest_loadpath should contain the path
    to the latest restoration point. This is automatically set in the framework
    :param net: Network to restore weights of
    :param optimizer: sometimes the optimizer also needs to be restored.
    :param args:
    :return: Nothing, only restore the network and optimizer.
    """

    restore_path = '{}'.format(args.latest_loadpath)
    print('Latest, continuing from {}'.format(restore_path))
    checkpoint = torch.load(restore_path, map_location=lambda storage, loc: storage)

    for name, field in restore_fields.items():
        new_state_dict = OrderedDict()
        for k, v in checkpoint[name].items():
            if 'module' in k and args.device == 'cpu':
                name = k.replace("module.", "")  # remove module.
            else:
                name = k
            new_state_dict[name] = v

        field.load_state_dict(new_state_dict)


def build_experiment_folder(experiment_name, log_path, save_images):
    """
    An experiment logging folder goes along with each experiment. This builds that folder
    :param args: dictionary of arguments
    :return: filepaths for saved models, logs, and images
    """
    saved_models_filepath = os.path.join(log_path, experiment_name.replace("%.%", "/"), "saved_models")
    logs_filepath = os.path.join(log_path, experiment_name.replace("%.%", "/"), "summary_logs")
    images_filepath = os.path.join(log_path, experiment_name.replace("%.%", "/"), "images")

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)

    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    if not os.path.exists(images_filepath):
        os.makedirs(images_filepath)

    if save_images:
        if not os.path.exists(images_filepath + '/train'):
            os.makedirs(images_filepath + '/train')
        if not os.path.exists(images_filepath + '/test'):
            os.makedirs(images_filepath + '/test')

    return saved_models_filepath, logs_filepath, images_filepath


def get_start_epoch(args):
    """
    Utility for finding where to resume from
    :param args: args parsed from input
    :return: where to start resume
    """
    latest_loadpath = ''
    latest_trained_epoch = -1
    if args.resume:
        from glob import glob
        # search for the latest
        saved_ckpts = glob('{}/*checkpoint.pth.tar'.format(args.saved_models_filepath))
        if len(saved_ckpts) == 0:
            args.resume = 0
        else:
            for fn in saved_ckpts:
                if 'best' not in fn.split('/')[-1]:
                    query_trained_epoch = int(fn.split('/')[-1].split('_')[0])
                    if query_trained_epoch > latest_trained_epoch:
                        latest_trained_epoch = query_trained_epoch
                        latest_loadpath = fn
    start_epoch = 0
    if args.resume:
        start_epoch = latest_trained_epoch + 1

    return start_epoch, latest_loadpath


def get_best_performing_epoch_on_target_metric(metrics_dict, target_metric, ranking_method=np.argmax):
    """
    utility for finding best epoch thus far
    :param: metrics_dict: A dictionary containing the collected metrics from which to extract the best perf. model epoch
    target_metric:
    ranking_method:
    :return: best epoch, and what the best target metric value was
    """
    best_model_epoch = 0
    best_target_metric = None

    if target_metric in metrics_dict:
        if len(metrics_dict[target_metric]) != 0:
            best_epoch_idx = ranking_method(metrics_dict[target_metric])
            best_model_epoch, best_target_metric = metrics_dict['epoch'][best_epoch_idx], \
                                                   metrics_dict[target_metric][best_epoch_idx]

    return best_model_epoch, best_target_metric


def print_network_stats(net):
    """
    Utility for printing how many parameters and weights in the network
    :param net: network to observe
    :return: nothing, just print
    """
    trainable_params_count = 0
    trainable_weights_count = 0
    for param in net.parameters():
        weight_count = 1
        for w in param.shape:
            weight_count *= w
        if param.requires_grad:
            trainable_params_count += 1
            trainable_weights_count += weight_count

    print('{} parameters and {} weights are trainable'.format(trainable_params_count, trainable_weights_count))
