import hfai
from torchvision import transforms
from .statistic import *


def create_transform(split):
    if split == "train":
        crop = transforms.RandomCrop(256)
    elif split == "val":
        crop = transforms.CenterCrop(256)
    else:
        raise ValueError(f"Unknown split: {split}")

    transform = transforms.Compose([
        transforms.Resize(256),
        crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transform


class ImageNet(hfai.datasets.ImageNet):

    def __init__(self, split) -> None:
        super().__init__(split)
        self.img_transform = create_transform(split)

    def __getitem__(self, indices):
        samples = super().__getitem__(indices)

        new_samples = []
        for img, label in samples:
            img = self.img_transform(img.convert("RGB"))
            new_samples.append(img)

        return new_samples


def coco(split):
    img_transform = create_transform(split)
    def transform(img, img_id, anno):
        img = img_transform(img)
        return img

    dataset = hfai.datasets.CocoCaption(split, transform=transform)
    return dataset


def googlecc(split):
    img_transform = create_transform(split)
    def transform(img, text):
        return img_transform(img.convert("RGB"))

    dataset = hfai.datasets.GoogleConceptualCaption(split, transform)
    return dataset


def imagenet(split):
    return ImageNet(split)
