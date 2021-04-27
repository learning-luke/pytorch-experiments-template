import os
import sys
from copy import copy
import argparse
from rich import print

experiment_json_dir = "../../experiment_files/experiment_config_files/"
main_experiment_script = "train.py"

# uncomment the ones needed
cluster_scripts = {
    # "apollo_cluster_multi": "cluster_multi_gpu_apollo_template_script",
    # "charles_cluster_multi": "cluster_multi_gpu_charles_template_script",
    # "apollo_cluster_single": "cluster_single_gpu_apollo_template_script",
    # "charles_cluster_single": "cluster_single_gpu_charles_template_script",
    "gpu_box": "gpu_box_template_script",
}
local_script_dir = "../../experiment_files/local_experiment_scripts"
cluster_script_dir = "../../experiment_files/cluster_experiment_scripts"

if not os.path.exists(local_script_dir):
    os.makedirs(local_script_dir)

if not os.path.exists(cluster_script_dir):
    os.makedirs(cluster_script_dir)

for subdir, dir, files in os.walk(local_script_dir):
    for file in files:
        if file.endswith(".sh"):
            filepath = os.path.join(subdir, file)
            os.remove(filepath)

for subdir, dir, files in os.walk(cluster_script_dir):
    for file in files:
        if file.endswith(".sh"):
            filepath = os.path.join(subdir, file)
            os.remove(filepath)


def load_template(filepath):
    with open(filepath, mode="r") as filereader:
        template = filereader.readlines()

    return template


def fill_template(template_list, execution_script, experiment_config):
    template_list = copy(template_list)
    execution_line = template_list[-1]
    execution_line = execution_line.replace("$execution_script$", execution_script)
    execution_line = execution_line.replace("$experiment_config$", experiment_config)
    template_list[-1] = execution_line
    return "".join(template_list)


def write_text_to_file(text, filepath):
    with open(filepath, mode="w") as filewrite:
        filewrite.write(text)


local_script_template = load_template("script_templates/local_run_template_script.sh")

for subdir, dir, files in os.walk(experiment_json_dir):
    for file in files:
        if file.endswith(".json"):
            config = file

            experiment_script = main_experiment_script

            for name, cluster_script_template_f in cluster_scripts.items():
                cluster_script_template = load_template(
                    "script_templates/{}.sh".format(cluster_script_template_f)
                )
                cluster_script_text = fill_template(
                    template_list=cluster_script_template,
                    execution_script=experiment_script,
                    experiment_config=file,
                )
                cluster_script_name = "{}/{}_{}.sh".format(
                    cluster_script_dir, file.replace(".json", ""), name
                ).replace(' ', '')
                cluster_script_name = os.path.abspath(cluster_script_name)

                write_text_to_file(cluster_script_text, filepath=cluster_script_name)

            local_script_text = fill_template(
                template_list=local_script_template,
                execution_script=experiment_script,
                experiment_config=file,
            )

            local_script_name = "{}/{}.sh".format(
                local_script_dir, file.replace(".json", "")
            )
            local_script_name = os.path.abspath(local_script_name)
            write_text_to_file(text=local_script_text, filepath=local_script_name)
            print(local_script_name)
