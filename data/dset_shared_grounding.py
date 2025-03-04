import os
import cv2
import pdb
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import sys
sys.path.append('.')
from data.template import grounding_to_qwen, batch_add_answer, batch_add_answer_append

dataset_mapping = {
    "showui-desktop": "ShowUI-desktop",
    "showui-web": "ShowUI-web",
    "amex": "AMEX",
    "rico": "RICO",
    "ricosca": "RICO",
    "widget": "RICO",
}

class GroundingDataset(torch.utils.data.Dataset):
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
        self.sample_prob = np.array([args_dict.get('text2point', 1), 
                                        args_dict.get('text2bbox', 0), 
                                        args_dict.get('point2text', 0), 
                                        args_dict.get('bbox2text', 0)])
        self.sample_prob = self.sample_prob / self.sample_prob.sum()
        self.random_sample = args_dict.get('random_sample', False)
        
        self.num_turn = args_dict.get('num_turn', 1)
        self.shuffle_image_token = args_dict.get('shuffle_image_token', False)
        self.uniform_prompt = args_dict.get('uniform_prompt', False)
        self.crop_min = args_dict.get('crop_min', 1)
        self.crop_max = args_dict.get('crop_max', 1)
        self.xy_int = args_dict.get('xy_int', False)

        print(f"Dataset: {dataset}; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        if self.random_sample:
            return self.samples_per_epoch
        else:
            return len(self.json_data)

    def __getitem__(self, idx):
        while True:
            try:
                return self.get_sample(idx)
            except Exception as e:
                # this is acceptable during training
                print(e)
                idx = random.randint(0, len(self.json_data) - 1)

    def random_crop_metadata(self, img, metadata, scale_range=(0.5, 1.0)):
        original_width, original_height = metadata['img_size']
        img_copy = img.copy()
        
        scale_w = random.uniform(*scale_range)
        scale_h = random.uniform(*scale_range)

        crop_width = int(original_width * scale_w)
        crop_height = int(original_height * scale_h)

        pad_x = pad_y = 0

        if crop_width > original_width or crop_height > original_height:
            pad_x = max(0, (crop_width - original_width) // 2)
            pad_y = max(0, (crop_height - original_height) // 2)

            padded_img = Image.new('RGB', (crop_width, crop_height), (255, 255, 255))
            padded_img.paste(img_copy, (pad_x, pad_y))

            img = padded_img
            img_width, img_height = crop_width, crop_height
        else:
            img_width, img_height = original_width, original_height

        crop_x_min = random.randint(0, img_width - crop_width)
        crop_y_min = random.randint(0, img_height - crop_height)
        crop_x_max = crop_x_min + crop_width
        crop_y_max = crop_y_min + crop_height

        cropped_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

        new_elements = []
        for element in metadata['element']:
            bbox = element['bbox']
            point = element['point']

            bbox_abs = [int(bbox[0] * original_width) + pad_x, int(bbox[1] * original_height) + pad_y,  
                        int(bbox[2] * original_width) + pad_x, int(bbox[3] * original_height) + pad_y]
            point_abs = [int(point[0] * original_width) + pad_x, int(point[1] * original_height) + pad_y]

            if (bbox_abs[0] >= crop_x_min and bbox_abs[2] <= crop_x_max and
                bbox_abs[1] >= crop_y_min and bbox_abs[3] <= crop_y_max):
                
                new_bbox = [(bbox_abs[0] - crop_x_min) / crop_width,
                            (bbox_abs[1] - crop_y_min) / crop_height,
                            (bbox_abs[2] - crop_x_min) / crop_width,
                            (bbox_abs[3] - crop_y_min) / crop_height]
                new_point = [(point_abs[0] - crop_x_min) / crop_width,
                            (point_abs[1] - crop_y_min) / crop_height]

                new_element = element.copy()
                new_element['bbox'] = new_bbox
                new_element['point'] = new_point
                new_elements.append(new_element)

        if len(new_elements) == 0:
            return img_copy, metadata

        metadata['element'] = new_elements
        metadata['element_size'] = len(new_elements)
        metadata['img_size'] = cropped_img.size
        return cropped_img, metadata

    def get_sample(self, idx):
        if not self.inference and self.random_sample:
            idx = random.randint(0, len(self.json_data) - 1)
        idx = idx % len(self.json_data)
        
        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"])
            image_list = [Image.open(image_path).convert("RGB")]
            if self.crop_min != 1 or self.crop_max != 1:
                image_list[0], item = self.random_crop_metadata(image_list[0], item, scale_range=(self.crop_min, self.crop_max))
        else:
            image_path = ""
            image_list = None

        # text2point, point2text, text2bbox, bbox2text
        sample_io = np.random.choice(len(self.sample_prob), p=self.sample_prob)

        # prepare for multi-turn streaming
        element_list = item['element']
        k = min(self.num_turn, len(element_list))
        assert k > 0
        element_cand = random.choices(element_list, k=k)

        if len(element_cand) == 1:
            element = element_cand[0]
            element_name = element['instruction']
            answer_xy = element['point'] if sample_io in [0, 2] else element['bbox']
            if self.xy_int:
                answer_xy = [int(x * 1000) for x in answer_xy]
            else:
                answer_xy = [round(x, 2) for x in answer_xy]
            if sample_io in [2, 3]:
                element_name, answer_xy = answer_xy, element_name

            img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
            source = grounding_to_qwen(element_name, img_dict, sample_io, self.shuffle_image_token, self.xy_int, self.uniform_prompt)
            prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
            data_dict_q = self.processor(text=prompt, images=image_list, return_tensors="pt", training=not self.inference)
            data_dict_qa, answer = batch_add_answer(data_dict_q, answer_xy, self.processor)
            
        else:
            # multi-turn streaming
            element_name_list = [element['instruction'] for element in element_cand]
            if sample_io in [0, 2]:
                answer_xy_list = [element['point'] for element in element_cand]
            else:
                answer_xy_list = [element['bbox'] for element in element_cand]

            if self.xy_int:
                answer_xy_list = [[int(x * 1000) for x in answer_xy] for answer_xy in answer_xy_list]
            else:
                answer_xy_list = [[round(x, 2) for x in answer_xy] for answer_xy in answer_xy_list]

            if sample_io in [2, 3]:
                element_name_list, answer_xy_list = answer_xy_list, element_name_list
            answer_xy = answer_xy_list[0]

            img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
            source = grounding_to_qwen(element_name_list[0], img_dict, sample_io, self.shuffle_image_token, self.xy_int, self.uniform_prompt)
            prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True) 
            data_dict_q = self.processor(text=prompt, images=image_list, return_tensors="pt", training=not self.inference)
            data_dict_qa, answer = batch_add_answer_append(data_dict_q, answer_xy, self.processor,
                                                            append_element=element_name_list[1:], 
                                                            append_answer=answer_xy_list[1:])

        max_seq_len = self.processor.tokenizer.model_max_length
        data_dict = dict(
            input_ids=data_dict_qa["input_ids"][0],
            image_sizes=data_dict_qa["image_grid_thw"],
            pixel_values=data_dict_qa["pixel_values"],
            labels=data_dict_qa["labels"][0],        
        )
        assert data_dict_qa["input_ids"][0].shape[0] <= max_seq_len, f"Input seq. surpasses Max. seq len: {data_dict_qa['input_ids'][0].shape[0]} > {max_seq_len}"

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
                                            uigraph_train=True, uigraph_test=True,
                                            uigraph_diff=1,  uigraph_rand=False,
                                            uimask_pre=True, uimask_ratio=1, uimask_rand=False
                                            )

    dataset = GroundingDataset(
        "/blob/v-lqinghong/data/GUI_database",
        "rico",
        "hf_train_rico",
        processor,
        inference=False
    )

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        data_size = str(data[1]['img_size'])
        print(i, len(data[0]['input_ids']), data[0]['patch_assign_len'], data[0]['select_mask'].sum())
