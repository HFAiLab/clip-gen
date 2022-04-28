import hfai
import torchvision.transforms.functional as TF
from torchvision import transforms
from .statistic import *


class ImageTransform():

    def __init__(self, split) -> None:
        if split == "train":
            self.crop = transforms.RandomCrop(256)
        elif split == "val":
            self.crop = transforms.CenterCrop(256)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.normalize1 = transforms.Normalize(mean=mean, std=std)
        self.normalize2 = transforms.Normalize(mean=clip_mean, std=clip_std)

    def __call__(self, img):
        img = TF.resize(img, 256)
        img = self.crop(img)

        # to VQGAN, size (256, 256)
        img1 = TF.to_tensor(img)
        img1 = self.normalize1(img1)

        # to CLIP, size (224, 224)
        img2 = TF.resize(img, 224)
        img2 = TF.to_tensor(img2)
        img2 = self.normalize2(img2)

        return img1, img2


class ImageNet(hfai.datasets.ImageNet):

    def __init__(self, split) -> None:
        super().__init__(split)
        self.img_transform = ImageTransform(split)

    def __getitem__(self, indices):
        samples = super().__getitem__(indices)

        new_samples = []
        for img, _ in samples:
            img1, img2 = self.img_transform(img.convert("RGB"))
            new_samples.append((img1, img2))

        return new_samples


def coco(split):
    img_transform = ImageTransform(split)

    def transform(img, img_id, anno):
        return img_transform(img)

    dataset = hfai.datasets.CocoCaption(split=split, transform=transform)
    return dataset


def googlecc(split):
    img_transform = ImageTransform(split)
    def transform(img, text):
        img1, img2 = img_transform(img.convert("RGB"))
        return img1, img2

    return hfai.datasets.GoogleConceptualCaption(split, transform)


def imagenet(split):
    return ImageNet(split)
