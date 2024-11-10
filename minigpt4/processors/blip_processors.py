"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from minigpt4.common.registry import registry
from minigpt4.processors.base_processor import BaseProcessor
from minigpt4.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size,image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )

import random
import torch

@registry.register_processor("robustvlguard_image_train")
class RobustVLGuardBlip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self,
        image_size=224,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        apply_gaussian_noise_prob=0.7,
        noise_std_range=(0.05, 0.15)
    ):
        super().__init__(mean=mean, std=std)
        
        # 设置高斯噪声参数
        self.apply_gaussian_noise_prob = apply_gaussian_noise_prob
        self.noise_std_range = noise_std_range

        # 图像变换操作
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.add_gaussian_noise_if_applicable,  # 添加高斯噪声增强方法
                self.normalize,
            ]
        )

    def add_gaussian_noise_if_applicable(self, image):
        """随机决定是否对图像添加高斯噪声"""
        if random.random() < self.apply_gaussian_noise_prob:
            return self.add_gaussian_noise(image)
        return image

    def add_gaussian_noise(self, image):
        """对图像添加高斯噪声"""
        std_dev = random.uniform(*self.noise_std_range)  # 随机选择噪声的标准差
        noise = torch.normal(0, std_dev, image.shape)  # 生成噪声张量
        noisy_image = torch.clamp(image + noise, 0, 1)  # 添加噪声并裁剪到 [0, 1] 范围
        return noisy_image

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)
        apply_gaussian_noise_prob = cfg.get("apply_gaussian_noise_prob", 0.7)
        noise_std_range = cfg.get("noise_std_range", (0.05, 0.15))

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            apply_gaussian_noise_prob=apply_gaussian_noise_prob,
            noise_std_range=noise_std_range,
        )


@registry.register_processor("blip2_image_eval")
class Blip2ImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)
