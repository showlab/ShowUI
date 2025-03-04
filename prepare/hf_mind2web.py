from PIL import Image, ImageDraw
import json
import os
import re
import pdb
from tqdm import tqdm
import random
import argparse

# imgs_dir =  "./datasets/GUI_database/Mind2Web/images"
# anno_dir = "./datasets/GUI_database/Mind2Web/metadata"

imgs_dir =  "/blob/v-lqinghong/data/GUI_database/Mind2Web/images"
anno_dir = "/blob/v-lqinghong/data/GUI_database/Mind2Web/metadata"

def data_transform(version='train', mini=False):
    mind2web_train = json.load(open(f"{anno_dir}/mind2web_data_{version}.json", 'r'))

    total_step = []
    step_i = 0

    for episode in tqdm(mind2web_train):
        annot_id = episode["annotation_id"]
        confirmed_task = episode["confirmed_task"]

        step_history = []
        repr_history = []
        for i, (step, step_repr) in enumerate(zip(episode["actions"], episode["action_reprs"])):
            filename = annot_id + '-' + step["action_uid"] + '.jpg'
            img_path = os.path.join(imgs_dir, filename)
            
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path)

            total_step.append({
                            "split": version,
                            "id": "mind2web_{}".format(step_i), 
                            "annot_id": annot_id,
                            "action_uid": step["action_uid"],
                            
                            "website": episode["website"],
                            "domain": episode["domain"],
                            "subdomain": episode["subdomain"],

                            "task": confirmed_task,
                            "img_url": filename,
                            "img_size": image.size,

                            "step_id": i,
                            "step": step,
                            "step_repr": step_repr,
                            "step_history": step_history.copy(),
                            "repr_history": repr_history.copy()
                            })

            step_history.append(step)
            repr_history.append(step_repr)

            step_i += 1
            
        if mini and step_i > 1:
            break

    if mini:
        return total_step

    save_url = f"{anno_dir}/hf_{version}.json"
    with open(save_url, "w") as file:
        json.dump(total_step, file, indent=4)
    return total_step

if __name__ == "__main__":
    for version in ['train']:
        data_transform(version=version)

    test_full = []
    for version in ['test_task', 'test_domain', 'test_website']:
        test_full.extend(data_transform(version=version))
    save_url = f"{anno_dir}/hf_test_full.json"
    with open(save_url, "w") as file:
        json.dump(test_full, file, indent=4)

    # miniset
    # test_full = []
    # for version in ['test_task', 'test_domain', 'test_website']:
    #     test_full.extend(data_transform(version=version, mini=True))
    
    # save_url = f"{anno_dir}/hf_test_mini.json"
    # with open(save_url, "w") as file:
    #     json.dump(test_full, file, indent=4)
