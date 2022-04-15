import math
import os
import shutil
from pathlib import Path
import torch

try:
    import hfai
    import hfai.distributed as dist
except:
    pass


_print = print


class CLIPWrapper():

    def __init__(self, clip, normalize=True):
        self.clip = clip.eval()
        self.normalize = normalize
        if normalize:
            print("normalize CLIP embeddings")

    def encode_image(self, image):
        embeds = self.clip.encode_image(image)
        if self.normalize:
            embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds

    @torch.no_grad()
    def encode_text(self, text):
        embeds = self.clip.encode_text(text)
        if self.normalize:
            embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds


class CosineLRWarmUp:
    def __init__(self, optimizer, warmup_epochs, epochs, lr, min_lr, enabled=True):
        self.optimizer = optimizer
        self.wepochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr
        self.enabled = enabled

    def step(self, epoch):
        if not self.enabled:
            return self.lr

        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.wepochs:
            lr = self.lr * epoch / self.wepochs
        else:
            angle = math.pi * (epoch - self.wepochs) / (self.epochs - self.wepochs)
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (1.0 + math.cos(angle))

        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr


def configure_optimizer(gpt, lr, wd=0.01, beta1=0.9, beta2=0.95):
    decay = set()
    no_decay = set()
    whitelist = (torch.nn.Linear, torch.nn.MultiheadAttention, hfai.nn.MultiheadAttention)
    blacklist = (torch.nn.LayerNorm, torch.nn.Embedding, hfai.nn.LayerNorm)
    for mn, m in gpt.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add('pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in gpt.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    outer_params = param_dict.keys() - union_params
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert len(outer_params) == 0, f"parameters {outer_params} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(beta1, beta2))

    return optimizer


def save_model(state, filename):
    filename = str(filename)
    torch.save(state, filename + ".tmp")

    # rename
    if Path(filename).exists():
        Path(filename).rename(filename + ".old")

    Path(filename + ".tmp").rename(filename)

    if Path(filename + ".old").exists():
        Path(filename + ".old").unlink()


def init_dist(local_rank):
    # init dist
    ip = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    hosts = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size()


def backup(fname, save_path):
    if dist.get_rank() == 0:
        fname = Path(fname).absolute()
        dest = save_path / 'train.py'
        shutil.copyfile(fname, dest)
        print(f"copy {fname} -> {dest}")


def print(*args, **kwargs):
    if torch.cuda.current_device() == 0:
        _print(*args, **kwargs, flush=True)
