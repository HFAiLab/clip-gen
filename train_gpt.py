import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

import hfai
import hfai.distributed as dist
from hfai.nn.parallel import DistributedDataParallel

from models.vqgan import VQGAN
from models.clip import clip_vit_b32
from models.gpt import __dict__ as gpts
from datasets.gpt import __dict__ as datasets
from datasets.statistic import mean, std
from utils import *


###########################################
# CONFIG
###########################################

parser = argparse.ArgumentParser(description="Train GPT")
parser.add_argument("--ds", type=str, default="coco", help="dataset name")
parser.add_argument("--gpt", type=str, default="gpt2_medium", help="GPT model")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--bs", type=int, default=2, help="batch size")
parser.add_argument("--vqgan_ckpt", type=str, help="VQGAN pretrained model")
args = parser.parse_args()

gpt_name = args.gpt
dataset_name = args.ds
batch_size = args.bs
dropout = args.dropout
vqgan_ckpt = args.vqgan_ckpt

codebook_size = 16384
embed_dim = 256
normalize_clip = True
loss_clip_weight = 0

use_amp = False
enabled_warmup = True

epochs = 200
warmup_epochs = 20
min_lr = 0
base_lr = 5e-4 / 256

save_path = Path(f"output/gpt/{dataset_name}")

# sample
top_k = 500
top_p = 0.95

writer = None


def log_recon_images(name, x, x_recon, step):
    std1 = torch.tensor(std).view(1, -1, 1, 1).to(x)
    mean1 = torch.tensor(mean).view(1, -1, 1, 1).to(x)
    img = torch.cat([x_recon, x], dim=0)  # [2 * N, 3, H, W]
    img = img * std1 + mean1
    img = make_grid(img, nrow=x.size(0))
    writer.add_image(name, img.clamp(0, 1), step)


def logits_to_z(vqgan, logits):
    # logits, probs: size (B, L, vocab_size)
    probs = F.softmax(logits, dim=-1)
    embedding = vqgan.quantizer.embedding.weight  # (vocab_size, E)
    vocab_size, E = embedding.shape

    # argmax:  size (B, L)
    # one-hot: size (B, L, vocab_size)
    argmax = torch.argmax(logits, dim=2)
    onehot = F.one_hot(argmax, num_classes=vocab_size).float()

    # quantize
    onehot = probs + (onehot - probs).detach()

    B, L, vocab_size = onehot.shape
    z = onehot.view(B * L, vocab_size) @ embedding
    z = z.view(B, L, E)

    return z


def train(dataloader, gpt, vqgan, clip, optimizer, scheduler, scaler, epoch, start_step, best_score):
    gpt.train()
    steps_per_epoch = len(dataloader) + start_step

    for step, batch in enumerate(dataloader):
        step += start_step
        lr = scheduler.step(epoch + step / steps_per_epoch)
        x, clip_x = [t.cuda() for t in batch[:2]]  # images

        # prepare data
        with torch.no_grad():
            _, _, indices = vqgan.encode(x)
            indices = indices.view(indices.shape[0], -1).detach()  # [B, h * w]
            embeddings = clip.encode_image(clip_x)  # [B, 512]

        L = indices.size(1)
        input_tokens = indices[:, :(L - 1)].contiguous()  # [B, L]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = gpt(input_tokens, embeddings)

        loss_gpt = F.cross_entropy(logits.view(-1, logits.size(-1)), indices.view(-1))

        # CLIP embedding reconstruction loss
        if loss_clip_weight > 0:
            z = logits_to_z(vqgan, logits).view(-1, 16, 16, 256)
            z = z.permute(0, 3, 1, 2)
            x_recon = vqgan.decode(z)  # [B, 3, H, W]
            clip_x_recon = F.interpolate(x_recon, size=224, mode='bilinear')
            clip_embeds_recon = clip.encode_image(clip_x_recon)  # [B, 512]
            loss_clip = F.mse_loss(embeddings, clip_embeds_recon)
        else:
            loss_clip = torch.tensor(0).to(x)

        loss = loss_gpt + loss_clip_weight * loss_clip

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # save checkpoint if going to suspend
        rank = dist.get_rank()
        if rank == 0 and hfai.receive_suspend_command():
            state = {
                "model": gpt.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step + 1,
                "val_loss": best_score,
            }
            save_model(state, save_path / "latest.pt")
            print("going to suspend...")
            hfai.go_suspend()

        # log
        world_size = dist.get_world_size()
        for t in [loss, loss_gpt, loss_clip]:
            dist.all_reduce(t)
            t.div_(world_size)

        total_steps = epoch * steps_per_epoch + step
        if step % 10 == 0:
            mem_used = torch.cuda.max_memory_reserved() // (1 << 20)
            print(f"Epoch: {epoch}, Step: {step}, loss: {loss:.3f}, loss_gpt: {loss_gpt:.3f}, loss_clip: {loss_clip:.3f}, "
                  f"lr: {lr:.5f}, MemUsed: {mem_used} MiB")

            if rank == 0:
                writer.add_scalar("train/loss", loss, total_steps)
                writer.add_scalar("train/loss_gpt", loss_gpt, total_steps)
                writer.add_scalar("train/loss_clip", loss_clip, total_steps)
                writer.add_scalar("train/lr", lr, total_steps)

        if rank == 0 and total_steps % 200 == 0:
            # sample image
            z_idx = gpt.module.sample(embeddings, steps=16 * 16, top_k=top_k, top_p=top_p)  # [B, 16*16]
            with torch.no_grad():
                z_idx = z_idx.view(-1, 16, 16)
                z = vqgan.quantizer.decode(z_idx)  # (B, H, W, C)
                z = z.permute(0, 3, 1, 2)  # [B, C, H, W]
                x_recon = vqgan.decode(z)  # [B, 3, 256, 256]

            log_recon_images("train/from-images", x, x_recon, total_steps)


@torch.no_grad()
def validate(dataloader, gpt, vqgan, clip, epoch):
    gpt.eval()

    total, val_loss = torch.zeros(2).cuda()
    for batch in dataloader:
        x, clip_x = [t.cuda() for t in batch]

        _, _, indices = vqgan.encode(x)
        indices = indices.view(indices.shape[0], -1).detach()  # [B, h * w]
        embeddings = clip.encode_image(clip_x)     # [B, 512]

        L = indices.size(1)
        logits = gpt.module(indices[:, :(L - 1)], embeddings)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), indices.view(-1))

        val_loss += loss * x.shape[0]
        total += x.shape[0]

    for t in [total, val_loss]:
        dist.all_reduce(t)

    val_loss = val_loss.item() / total.item()
    print(f"=== Validate: epoch {epoch}, val_loss {val_loss:.3f}")

    if dist.get_rank() == 0:
        # decode the last batch
        z_idx = gpt.module.sample(embeddings, steps=256, top_k=top_k, top_p=top_p)  # [-1, 16*16]
        z_idx = z_idx.view(-1, 16, 16)
        z = vqgan.quantizer.decode(z_idx)  # (B, H, W, C)
        z = z.permute(0, 3, 1, 2)  # [B, C, H, W]
        x_recon = vqgan.decode(z)  # [B, 3, H, W]

        writer.add_scalar("val/val_loss", val_loss, epoch)
        log_recon_images("val/from-texts", x, x_recon, epoch)

    dist.barrier()
    return val_loss


def main(local_rank):
    log_path = save_path / "runs"
    save_path.mkdir(exist_ok=True, parents=True)
    rank, world_size = init_dist(local_rank)
    backup(__file__, save_path)

    # fix the seed for reproducibility
    torch.manual_seed(rank)

    if rank == 0:
        global writer
        writer = SummaryWriter(log_path)

    total_batch_size = batch_size * world_size
    lr = base_lr * total_batch_size

    ##################################
    # VQGAN
    ##################################
    vqgan = VQGAN(codebook_size, embed_dim).cuda().eval().requires_grad_(False)
    state = torch.load(vqgan_ckpt, map_location='cpu')
    vqgan.load_state_dict(state['model'])
    print(f"Loaded VQGAN model from {vqgan_ckpt}, epoch {state['epoch']}")

    ##################################
    # GPT
    ##################################
    gpt = gpts[gpt_name](vocab_size=codebook_size, dropout=dropout)
    gpt = DistributedDataParallel(gpt, device_ids=[local_rank])

    ##################################
    # CLIP
    ##################################
    clip = clip_vit_b32(pretrained=True).cuda().eval().requires_grad_(False)
    clip = CLIPWrapper(clip, normalize=normalize_clip)

    ##################################
    # datasets
    ##################################
    train_dataset = datasets[dataset_name]('train')
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = train_dataset.loader(batch_size, num_workers=8, sampler=train_sampler, pin_memory=True)

    val_dataset = datasets[dataset_name]('val')
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = val_dataset.loader(batch_size, num_workers=8, sampler=val_sampler, pin_memory=True, drop_last=True)

    ##################################
    # scaler & optimizer
    ##################################
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = configure_optimizer(gpt.module, lr)
    scheduler = CosineLRWarmUp(optimizer, warmup_epochs, epochs, lr, min_lr, enabled=enabled_warmup)

    # load
    best_score = torch.inf
    start_epoch, start_step = 0, 0
    latest_path = save_path / "latest.pt"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu")
        gpt.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        start_step = ckpt["step"]
        best_score = ckpt["val_loss"]
        print(f"loaded GPT model from epoch {start_epoch}, step {start_step}")
    else:
        print(f"{latest_path} not found, start training from scractch")

    # validate(val_loader, gpt, vqgan, clip, start_epoch - 1)

    # train, validate
    for epoch in range(start_epoch, epochs):
        # resume from epoch and step
        train_sampler.set_epoch(epoch)
        train_loader.set_step(start_step)

        train(train_loader, gpt, vqgan, clip, optimizer, scheduler, scaler, epoch, start_step, best_score)

        val_loss = torch.inf
        if epoch % 10 == 0 or epoch == epochs - 1:
            val_loss = validate(val_loader, gpt, vqgan, clip, epoch)

        start_step = 0  # reset
        # save
        if rank == 0:
            state = {
                "model": gpt.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "val_loss": min(best_score, val_loss),
                "step": 0,
            }
            save_model(state, latest_path)

            if epoch % 10 == 0 or epoch == epochs - 1:
                save_model(state, save_path / f"{epoch:04d}.pt")

            if val_loss < best_score:
                best_score = val_loss
                save_model(state, save_path / "best.pt")

    if writer:
        writer.close()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)

