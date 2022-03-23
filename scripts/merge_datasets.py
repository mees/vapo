import json
import os

import hydra

def get_abs_path(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


# Merge datasets using json files
def merge_datasets(directory_list, output_dir):
    output_dir = get_abs_path(output_dir)
    new_data = {"training": {}, "validation": {}}
    for dir in directory_list:
        abs_dir = get_abs_path(dir)
        json_path = os.path.join(abs_dir, "episodes_split.json")
        with open(json_path) as f:
            data = json.load(f)

        # Rename episode numbers if repeated
        data_keys = list(data.keys())
        split_keys = ["validation", "training"]
        other_keys = [k for k in data_keys if k not in split_keys]
        episode = 0
        for split in split_keys:
            dataset_name = os.path.basename(os.path.normpath(dir))
            for key in data[split].keys():
                new_data[split]["/%s/%s" % (dataset_name, key)] = data[split][key]
                episode += 1
        for key in other_keys:
            new_data[key] = data[key]
    # Write output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_file = os.path.join(output_dir, "episodes_split.json")
    with open(out_file, "w") as outfile:
        json.dump(new_data, outfile, indent=2)


@hydra.main(config_path="../config", config_name="cfg_merge_dataset")
def main(cfg):
    merge_datasets(cfg.data_lst, cfg.output_dir)


if __name__ == "__main__":
    main()
