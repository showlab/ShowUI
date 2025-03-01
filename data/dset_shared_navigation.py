import json
import os
import re
import pdb
import random

import cv2
from PIL import Image
import torch
import torch.nn.functional as F

import sys
sys.path.append('.')
from data.template import navigation_to_qwen, batch_add_answer
from qwen_vl_utils import process_vision_info

dataset_mapping = {
    "guiact": "GUI_Course/GUIAct",
}


"""
GUIAct action space
web-single: CLICK, INPUT, SELECT, HOVER, ANSWER, ENTER, SCROLL
web-multi: SCROLL, CLICK, HOVER, SELECT_TEXT, COPY, ANSWER, INPUT, ENTER
smartphone: SWIPE, TAP, INPUT, ANSWER, ENTER
"""

_SPLIT_MAP = {
    'hf_train_web-single': 'web-single',
    'hf_train_web-multi': 'web-multi',
    'hf_train_smartphone': 'smartphone',
    'hf_test_web-single': 'web-single',
    'hf_test_web-multi': 'web-multi',
    'hf_test_smartphone': 'smartphone',
}

def get_answer(step):
    action = step['action_type'].upper()
    action_value = step['action_value']
    action_point = step['point']

    click_point = None
    type_text = None

    if action in ['CLICK', 'INPUT', 'SELECT', 'HOVER', 'TAP']:
        click_point = action_point
        if click_point is not None:
            click_point = [round(item, 2) for item in click_point]
    elif action in ['SELECT_TEXT', 'SWIPE']:
        start_point = action_point[0]
        end_point = action_point[1]
        start_point = [round(item, 2) for item in start_point]
        end_point = [round(item, 2) for item in end_point]
        click_point = [start_point, end_point]

    if action in ['INPUT', 'SELECT', 'ANSWER', 'COPY', 'SCROLL']:
        type_text = action_value
    
    answer = {'action': action.upper(), 'value': type_text, 'position': click_point}
    return answer

def get_answer_by_list(step):
    if isinstance(step, dict):
        return get_answer(step)
    elif isinstance(step, list):
        tmp = [str(get_answer(item)) for item in step]
        tmp = ','.join(tmp)
        return tmp

def get_history(sample, num_history):
    step_history = sample['step_history']
    action_history = []
    for i, step in enumerate(step_history[-num_history:], start=1):
        action = get_answer(step)
        action_history.append(f'Step{i}: {action}')
    return '; '.join(action_history)

class NavigationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset,
        json_data,
        processor,
        inference=False,
        args_dict={},
    ):
        self.processor = processor
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels
        self.inference = inference

        self.base_image_dir = os.path.join(dataset_dir, dataset_mapping[dataset])
        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)

        self.samples_per_epoch = args_dict.get('samples_per_epoch', 1)

        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)
        self.num_turn = args_dict.get('num_turn', 0)
        self.num_history = args_dict.get('num_history', 0)
        self.interleaved_history = args_dict.get('interleaved_history', 'tttt')
        assert self.interleaved_history in ['tttt', 'vvvv', 'vtvt']
        self.random_sample = args_dict.get('random_sample', False)

        self.vis_start = self.processor.tokenizer('<|vision_start|>')['input_ids']
        self.vis_end = self.processor.tokenizer('<|vision_end|>')['input_ids']

        self.split = _SPLIT_MAP[json_data.replace('_v2', '')]
        print(f"Dataset: Mind2Web; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        # inference
        if self.inference:
            return len(self.json_data)

        # training
        if self.random_sample:
            return self.samples_per_epoch
        else:
            return len(self.json_data)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def append_history_image(self, sample, num_history, image_list, url_only=False):
        if num_history == 0:
            return image_list
        step_history = sample['step_history']
        for i, step in enumerate(step_history[-num_history:], start=1):
            image_path = os.path.join(self.IMG_DIR, step["img_url"]+'.png')
            if url_only:
                image_list.append(image_path)
            else:
                image_list.append(Image.open(image_path).convert("RGB"))
        return image_list

    def get_history_qwen(self, image_list, sample, num_history, interleaved_history='tttt', decay_factor=1):
        curr_image = image_list[-1]
        curr_dict = [{'type': 'image', 'image': curr_image, 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}]
        if num_history == 0 or sample['step_history'] == []:
            assert len(image_list) == 1
            return curr_dict

        step_history = sample['step_history']
        action_history = []
        action_prefix = []
        for i, step in enumerate(step_history[-num_history:]):
            # action = get_answer(step)
            action = get_answer_by_list(step['step']) if 'step' in step else get_answer(step)
            max_pixels = max(self.min_pixels, self.max_pixels * decay_factor ** (num_history - i))

            if interleaved_history == 'vvtt':
                action_prefix.append({"type": "image", "image": image_list[i], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
            elif interleaved_history == 'ttvv':
                action_prefix.append({"type": "text", "text": f'{action}'})

            if interleaved_history in ['tttt', 'vvtt']:
                action_history.append({"type": "text", "text": f'{action}'})
            elif interleaved_history in ['vvvv', 'ttvv']:
                action_history.append({"type": "image", "image": image_list[i], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
            elif interleaved_history == 'vtvt':
                action_history.append({"type": "image", "image": image_list[i], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
                action_history.append({"type": "text", "text": f'{action}'})
            elif interleaved_history == 'tvtv':
                action_history.append({"type": "text", "text": f'{action}'})
                action_history.append({"type": "image", "image": image_list[i], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
        tmp = action_prefix + action_history + curr_dict
        return tmp

    def __getitem__(self, idx):
        return self.get_sample(idx)

    def activate_history_labels(self, input_ids: torch.Tensor, labels: torch.Tensor, vis_end: list, vis_start: list):
        # for streaming training;
        # we replace the labels where satisfy <|vision_end|>xxx<|vision_start|> with the input_ids that satisfy the same position;
        assert input_ids.shape == labels.shape, "input_ids and labels must have the same shape"
        
        L = input_ids.shape[1]
        vis_end_len = len(vis_end)
        vis_start_len = len(vis_start)
        
        i = 0
        while i <= L - (vis_end_len + vis_start_len):
            if torch.equal(input_ids[0, i:i+vis_end_len], torch.tensor(vis_end)):
                for j in range(i + vis_end_len, L - vis_start_len + 1):
                    if torch.equal(input_ids[0, j:j+vis_start_len], torch.tensor(vis_start)):
                        labels[0, i+vis_end_len:j] = input_ids[0, i+vis_end_len:j]
                        i = j + vis_start_len - 1
                        break
            i += 1
        return labels

    def get_sample(self, idx):
        if not self.inference and self.random_sample:
            idx = random.randint(0, len(self.json_data) - 1)
        idx = idx % len(self.json_data)

        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"]+'.png')
            image_list = [image_path]
        else:
            image_path = ""
            image_list = None
        item['img_url_abs'] = image_path

        if self.interleaved_history in ['vvvv', 'vvtt', 'ttvv', 'vtvt', 'tvtv']:
            image_list = self.append_history_image(item, self.num_history, image_list, url_only=True)
        image_list.append(image_list.pop(0))

        task = item['task']
        answer_dict = get_answer_by_list(item['step'])
        action_history = self.get_history_qwen(image_list, item, self.num_history, self.interleaved_history) # if item['step_history'] != [] else image_list

        item['anno_id'] = idx
        item['answer'] = answer_dict

        source = navigation_to_qwen(task, None, action_history, self.split, None)

        prompt = self.processor.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(source)
        data_dict_q = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt",
                                training=not self.inference)

        if self.inference:
            if 'labels' not in data_dict_q:
                data_dict_q['labels'] = data_dict_q['input_ids']

            data_dict = dict(
                input_ids=data_dict_q["input_ids"][0],
                pixel_values=data_dict_q["pixel_values"],
                image_sizes=data_dict_q["image_grid_thw"],
                labels=data_dict_q["labels"][0],
            )

            # Prepare elements for ShowUI
            for key in ['select_mask', 'patch_pos', 'patch_assign', 'patch_assign_len']:
                if key in data_dict_q:
                    data_dict[key] = data_dict_q[key]

            return (
                data_dict,
                item,
        )

        data_dict_qa, answer = batch_add_answer(data_dict_q, answer_dict, self.processor)

        if self.num_turn > 1:
            data_dict_qa["labels"] = self.activate_history_labels(data_dict_qa["input_ids"], data_dict_qa["labels"], self.vis_end, self.vis_start)

        data_dict = dict(
            input_ids=data_dict_qa["input_ids"][0],
            pixel_values=data_dict_qa["pixel_values"],
            image_sizes=data_dict_qa["image_grid_thw"],
            labels=data_dict_qa["labels"][0]
        )

        # Prepare elements for ShowUI
        for key in ['select_mask', 'patch_pos', 'patch_assign', 'patch_assign_len']:
            if key in data_dict_q:
                data_dict[key] = data_dict_q[key]

        return (
            data_dict,
            item,
        )

if __name__ == '__main__':
    from model.showui.processing_showui import ShowUIProcessor
    from model.showui.modeling_showui import ShowUIForConditionalGeneration

    processor = ShowUIProcessor.from_pretrained(
                                            "Qwen/Qwen2-VL-2B-Instruct", 
                                            min_pixels=1024*28*28, 
                                            max_pixels=1024*28*28,
                                            model_max_length=4096,
                                            uigraph_train=False, uigraph_test=False,
                                            uigraph_diff=1,  uigraph_rand=False,
                                            uimask_pre=True, uimask_ratio=1, uimask_rand=False
                                            )

    dataset = NavigationDataset(
        "/blob/v-lqinghong/data/GUI_database",
        "guiact",
        "hf_train",
        processor,
        inference=False,
        args_dict={'num_history': 2, 'interleaved_history': 'tttt'}
    )

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        data_size = str(data[1]['img_size'])
        print(i, len(data[0]['input_ids']), data[0]['patch_assign_len'], data[0]['select_mask'].sum())
