"""
Storage associated utilities
"""
import numpy as np
import scipy.misc
import shutil
import torch
from collections import OrderedDict
import scipy


import csv


def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a) if a != float('Inf') else 9999999
    except ValueError:
        return False
    else:
        return a == b


def read_results(filename):
    """
    Reads the results from the log file
    :param filename: log filename
    :return: results in a dict
    """
    results = {}
    headers = []
    with open(filename, 'r') as csvfile:
        resultsreader = csv.reader(csvfile)
        for i, row in enumerate(resultsreader):
            if i == 0:
                for col in row:
                    results[col] = []
                    headers.append(col)  # to keep track of an ordered addition of the headings
            else:
                for i, col in enumerate(row):
                    results[headers[i]].append(int(float(col)) if isint(col) else float(col))
    return results


def save_statistics(log_dir, statistics_file_name, list_of_statistics, create=False):
    """
    Saves a statistics .csv file with the statistics
    :param log_dir: Directory of log
    :param statistics_file_name: Name of .csv file
    :param list_of_statistics: A list of statistics to add in the file
    :param create: If True creates a new file, if False adds list to existing
    """
    if create:
        with open("{}/{}.csv".format(log_dir, statistics_file_name), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(list_of_statistics)
    else:
        with open("{}/{}.csv".format(log_dir, statistics_file_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list_of_statistics)


def load_statistics(log_dir, statistics_file_name):
    """
    Loads the statistics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param statistics_file_name: The name of the statistics file
    :return: A dict with the statistics
    """
    data_dict = dict()
    with open("{}/{}.csv".format(log_dir, statistics_file_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n", "").replace("\r", "").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n", "").replace("\r", "").split(",")
            for key, item in zip(data_labels, data):
                if item not in data_labels:
                    data_dict[key].append(item)
    return data_dict


def save_image_batch(filename, images, clip=True, types=('features', 'gray'), norm_means=(0.5, 0.5, 0.5), norm_stds=(0.5, 0.5, 0.5)):
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
    height = int(np.ceil(n/width))

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
            img = np.transpose(img.detach().cpu().numpy(), (1,2,0))
            if type_repeat == 'colour':
                img[:,:,0] = img[:,:,0] * norm_stds[0] + norm_means[0]
                img[:,:,1] = img[:,:,1] * norm_stds[1] + norm_means[1]
                img[:,:,2] = img[:,:,2] * norm_stds[2] + norm_means[2]
                img = np.clip(img, 0, 1)
            elif type_repeat == 'gray':
                img = np.clip(img, 0, 1)
            elif type_repeat == 'features':
                img -= minimum
                img /= maximum
                img = colormap(img)
                img = img[:,:,0][:,:,0:3]
            output_img[buffer + x * (img_w + buffer):buffer + (x) * (img_w + buffer) + img_w, actual_height * repeat + buffer + y * (img_h + buffer): actual_height * repeat + buffer + y * (img_h + buffer) + img_h, :] = img

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
    inputs_extended[:inputs.shape[0],:,:,:] = inputs

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


def restore_model(net, optimizer, args):
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

    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        if 'module' in k and args.device == 'cpu':
            name = k.replace("module.", "")  # remove module.
        else:
            name = k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    for k, v in checkpoint['optimizer'].items():
        if 'cuda' in k and args.device == 'cpu':
            name = k.replace("cuda.", "")  # remove module.
        else:
            name = k
        new_state_dict[name] = v

    optimizer.load_state_dict(new_state_dict)


def build_experiment_folder(args):
    """
    An experiment logging folder goes along with each experiment. This builds that folder
    :param args: dictionary of arguments
    :return: filepaths for saved models, logs, and images
    """
    experiment_name, log_path = args.exp_name, args.logs_path
    saved_models_filepath = "{}/{}/{}".format(log_path, experiment_name.replace("%.%", "/"), "saved_models")
    logs_filepath = "{}/{}/{}".format(log_path, experiment_name.replace("%.%", "/"), "summary_logs")
    images_filepath = "{}/{}/{}".format(log_path, experiment_name.replace("%.%", "/"), "images")

    import os

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)
    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)
    if not os.path.exists(images_filepath):
        os.makedirs(images_filepath)
    if args.save_images:
        if not os.path.exists(images_filepath + '/train'):
            os.makedirs(images_filepath + '/train')
        if not os.path.exists(images_filepath + '/test'):
            os.makedirs(images_filepath + '/test')

    args.saved_models_filepath = saved_models_filepath
    args.logs_filepath = logs_filepath
    args.images_filepath = images_filepath
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


def get_best_epoch(args):
    """
    utility for finding best epoch thus far
    :param args: args parsed from input
    :return: best epoch, and what the acc was
    """
    best_epoch = -1
    # Calculate the best loss from the results statistics if restarting
    best_test_acc = 0.0
    if args.resume:
        print("Checking {}/{}.csv".format(args.logs_filepath, "result_summary_statistics"))
        results = read_results("{}/{}.csv".format(args.logs_filepath, "result_summary_statistics"))
        maxi = np.argmax(results['test_acc'])
        best_test_acc = results['test_acc'][maxi]
        best_epoch = results['epoch'][maxi]
    return best_epoch, best_test_acc


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

    print('{} paramaters and {} weights are trainable'.format(trainable_params_count, trainable_weights_count))
