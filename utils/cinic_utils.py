import os
import glob
from shutil import copyfile
from rich import print

import errno


def download_url(url, root, filename):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # downloads file
    if os.path.isfile(fpath):
        print("Using downloaded file: " + fpath)
        return 1
    else:
        try:
            return _extracted_from_download_url_21("Downloading ", url, fpath, urllib)
        except:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                return _extracted_from_download_url_21(
                    "Failed download. Trying https -> http instead." " Downloading ",
                    url,
                    fpath,
                    urllib,
                )


def _extracted_from_download_url_21(arg0, url, fpath, urllib):
    print(arg0 + url + " to " + fpath)
    urllib.request.urlretrieve(url, fpath)
    return 0


def download_cinic(root):
    import tarfile

    filename = "CINIC-10.tar.gz"

    downloaded = download_url(
        "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz",
        root,
        filename,
    )

    # extract file
    if downloaded == 0:
        _extracted_from_download_cinic_14(tarfile, root, filename)


def _extracted_from_download_cinic_14(tarfile, root, filename):
    cwd = os.getcwd()
    tar = tarfile.open(os.path.join(root, filename), "r:gz")
    os.chdir(root)
    tar.extractall()
    tar.close()
    os.chdir(cwd)


def extend_cinic_10(cinic_dir="../data/Cinic-10", symlink=True):
    """
    Create enlarged cinic-10 from train and validation sets
    :param cinic_dir: Where to find cinic-10
    :param symlink: create symlinks? If false, copy.
    :return: Nothing just do
    """

    print("Creating and/or checking enlarged CINIC-10")
    cinic_directory = cinic_dir
    enlarge_directory = cinic_dir + "-enlarged"
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    sets = ["train", "valid", "test"]
    if not os.path.exists(enlarge_directory):
        os.makedirs(enlarge_directory)
    if not os.path.exists(enlarge_directory + "/train"):
        os.makedirs(enlarge_directory + "/train")
    if not os.path.exists(enlarge_directory + "/test"):
        os.makedirs(enlarge_directory + "/test")

    for c in classes:
        if not os.path.exists("{}/train/{}".format(enlarge_directory, c)):
            os.makedirs("{}/train/{}".format(enlarge_directory, c))
        if not os.path.exists("{}/test/{}".format(enlarge_directory, c)):
            os.makedirs("{}/test/{}".format(enlarge_directory, c))

    for s in sets:
        for c in classes:
            source_directory = "{}/{}/{}".format(cinic_directory, s, c)
            filenames = glob.glob("{}/*.png".format(source_directory))
            for fn in filenames:
                dest_fn = fn.split("/")[-1]
                if s in ["train", "valid"]:
                    dest_fn = "{}/train/{}/{}".format(enlarge_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(os.path.abspath(fn), os.path.abspath(dest_fn))
                    else:
                        if not os.path.isfile(dest_fn):
                            copyfile(fn, dest_fn)

                elif s == "test":
                    dest_fn = "{}/test/{}/{}".format(enlarge_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(os.path.abspath(fn), os.path.abspath(dest_fn))
                    else:
                        if not os.path.isfile(dest_fn):
                            copyfile(fn, dest_fn)
