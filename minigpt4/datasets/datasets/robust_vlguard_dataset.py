import os
import json
import pickle
import random
import time
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset
import torch

def custom_collate_fn(batch):
    # 对 "image" 字段的值使用 torch.stack 进行张量堆叠
    images = torch.stack([item["image"] for item in batch])
    
    # 对其他字段保留为字符串列表
    batch_dict = {
        "image": images,
        "instruction_input": [item["instruction_input"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "image_id": [item["image_id"] for item in batch]
    }
    
    return batch_dict

class RobustVLGuard_Dataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file (jsonl)
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.collater = custom_collate_fn
        self.repeat_factor = 2

        with open(ann_path, 'r') as f:
            self.ann = [json.loads(line) for line in f]

        self.ann = self.ann * self.repeat_factor
        # 随机打乱数据集
        random.shuffle(self.ann)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = info['image']
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        answer = info['conversations'][1]['value']
        instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()

        instruction = '<Img><ImageHere></Img> {} '.format(self.text_processor(instruction))

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['id'],
        }