import argparse
import time
from pathlib import Path
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import hfai
import hfai.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models.vqgan import VQGAN
from datasets.vqgan import __dict__ as datasets
from datasets.statistic import mean, std
from losses.lpips import LPIPS
from losses.hinge import hinge_d_loss, hinge_g_loss
from utils import save_model, init_dist, backup


###########################################
# CONFIG
###########################################

parser = argparse.ArgumentParser(description="Train VQ-GAN")
parser.add_argument("--ds", type=str, default="coco", help="dataset name")
parser.add_argument("--bs", type=int, default=2, help="batch size")
args = parser.parse_args()

# loss weights
perceptual_weight = 1.0
disc_weight = 0.8
codebook_weight = 1.0

disc_start_epochs = 1

# vqgan
codebook_size = 16384
embed_dim = 256

dataset_name = args.ds
batch_size = args.bs
epochs = 800
num_workers = 8
base_lr = 4.5e-6
save_path = Path(f"output/vqgan/{dataset_name}")

writer = None


def compute_adaptive_weight(model, loss_recon, loss_disc):
    last_layer = model.decoder.final[2].weight
    grad_disc = torch.autograd.grad(loss_disc, last_layer, retain_graph=True)[0]
    grad_recon = torch.autograd.grad(loss_recon, last_layer, retain_graph=True)[0]
    dist.all_reduce(grad_disc)
    dist.all_reduce(grad_recon)

    weight_d = torch.norm(grad_recon) / (torch.norm(grad_disc) + 1e-4)
    weight_d = torch.clamp(weight_d, 0.0, 1e4).item()
    return weight_d


@torch.no_grad()
def log_recons_image(name, x, x_recon, steps):
    std1 = torch.tensor(std).view(1, -1, 1, 1).cuda()
    mean1 = torch.tensor(mean).view(1, -1, 1, 1).cuda()
    x_recon = x_recon * std1 + mean1  # [B, C, H, W]
    x = x * std1 + mean1

    img = torch.cat([x, x_recon], dim=0).clamp(0, 1)
    img = make_grid(img, x.size(0))

    writer.add_image(name, img, steps)
    writer.flush()


def train(loader, model, lpips, opt_g, opt_d, epoch, local_rank, start_step, best_score):
    model.train()
    steps_per_epoch = len(loader) + start_step

    for step, x in enumerate(loader):
        step += start_step
        global_steps = epoch * steps_per_epoch + step
        x = x.cuda()  # images

        ##### stage 0: train E + G + Q
        opt_g.zero_grad()
        x_recon, loss_quant, logits_fake = model(x, stage=0)
        loss_l1 = (x - x_recon).abs().mean()
        loss_perceptual = lpips(x, x_recon).mean()

        loss_recon = loss_l1 + perceptual_weight * loss_perceptual
        loss_g = loss_recon + codebook_weight * loss_quant

        if epoch < disc_start_epochs:
            loss_disc = torch.tensor(0.).cuda()
            weight_d = torch.tensor(0.).cuda()
        else:
            loss_disc = hinge_g_loss(logits_fake)
            weight_d = compute_adaptive_weight(model.module, loss_recon, loss_disc)
            loss_g = loss_g + weight_d * disc_weight * loss_disc

        loss_g.backward()
        opt_g.step()

        ##### stage 1: train D

        if epoch < disc_start_epochs:
            loss_d = torch.tensor(0.).cuda()
        else:
            opt_d.zero_grad()
            logits_real, logits_fake = model(x, stage=1)
            loss_d = hinge_d_loss(logits_real, logits_fake)
            loss_d.backward()
            opt_d.step()

        # save checkpoint if going to suspend
        rank = dist.get_rank()
        if rank == 0 and hfai.receive_suspend_command():
            state = {
                "model": model.module.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch,
                "step": step + 1,
                "loss_recon": best_score,
            }
            save_model(state, save_path / "latest.pt")
            hfai.go_suspend()
            time.sleep(5)

        # log
        losses = torch.tensor([loss_g, loss_recon, loss_l1, loss_perceptual, loss_quant, loss_disc, loss_d]).cuda()
        dist.all_reduce(losses)
        losses /= dist.get_world_size()
        loss_g, loss_recon, loss_l1, loss_perceptual, loss_quant, loss_disc, loss_d = [v.item() for v in losses]

        if local_rank == 0 and step % 10 == 0:
            mem_used = torch.cuda.max_memory_reserved() // (1 << 20)
            print(f"Epoch: {epoch}, Step: {step}, loss_g: {loss_g:.3f}, loss_recon: {loss_recon:.3f}, "
                  f"loss_perceptual: {loss_perceptual:.3f}, loss_quant: {loss_quant:.3f}, "
                  f"weight_d: {weight_d:.3f}, MemUsed: {mem_used} MiB", flush=True)

        if rank == 0 and step % 10 == 0:
            writer.add_scalar("train/loss_g", loss_g, global_steps)
            writer.add_scalar("train/loss_recon", loss_recon, global_steps)
            writer.add_scalar("train/loss_perceptual", loss_perceptual, global_steps)
            writer.add_scalar("train/loss_quant", loss_quant, global_steps)
            writer.add_scalar("train/weight_d", weight_d, global_steps)

        if rank == 0 and step % 100 == 0:
            log_recons_image("train/img-recon", x, x_recon, global_steps)


@torch.no_grad()
def validate(loader, model, lpips, epoch, local_rank):
    model.eval()

    total, loss_l1s, loss_perceptuals, loss_recons = torch.zeros(4).cuda()
    for x in loader:
        x = x.cuda()  # images

        ##### autoencode
        x_recon, _, _ = model(x, stage=0)
        loss_l1 = (x - x_recon).abs().mean()
        loss_perceptual = lpips(x, x_recon).mean()
        loss_recon = loss_l1 + perceptual_weight * loss_perceptual

        loss_l1s += loss_l1 * x.shape[0]
        loss_perceptuals += loss_perceptual * x.shape[0]
        loss_recons += loss_recon * x.shape[0]
        total += x.shape[0]

    for t in [total, loss_l1s, loss_perceptuals, loss_recons]:
        dist.reduce(t, 0)

    loss_recon = loss_recons.item() / total.item()
    loss_l1 = loss_l1s.item() / total.item()
    loss_perceptual = loss_perceptuals.item() / total.item()

    if dist.get_rank() == 0:
        writer.add_scalar("val/loss_recon", loss_recon, epoch)
        writer.add_scalar("val/loss_perceptual", loss_perceptual, epoch)
        log_recons_image("val/img-recon", x, x_recon, epoch)

    if local_rank == 0:
        print(f"=== Validate: epoch {epoch}, loss_recon {loss_recon:.3f}, loss_l1 {loss_l1:.3f}, loss_perceptual {loss_perceptual:.3f}", flush=True)

    dist.barrier()
    return loss_recon


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

    ########################################
    # model
    ########################################
    module = VQGAN(codebook_size, embed_dim)
    model = DistributedDataParallel(module.cuda(), device_ids=[local_rank], find_unused_parameters=True)

    ########################################
    # dataloaders
    ########################################
    train_dataset = datasets[dataset_name]('train')
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = train_dataset.loader(batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    val_dataset = datasets[dataset_name]('val')
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = val_dataset.loader(batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)

    ########################################
    # optimizers
    ########################################

    # optimize E + G + Q
    g_params = list(module.encoder.parameters())   \
             + list(module.decoder.parameters())   \
             + list(module.quantizer.parameters())
    opt_g = torch.optim.Adam(g_params, lr=lr, betas=(0.5, 0.9))

    # optimize D
    d_params = module.discriminator.parameters()
    opt_d = torch.optim.Adam(d_params, lr=lr, betas=(0.5, 0.9))

    # perceptual loss
    lpips = LPIPS().cuda().eval()

    # load
    start_epoch, start_step, best_score = 0, 0, float('inf')
    if Path(save_path / "latest.pt").exists():
        ckpt = torch.load(save_path / "latest.pt", map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"]
        start_step = ckpt["step"]
        best_score = ckpt['loss_recon']
        print(f"loaded model from epoch {start_epoch}, step {start_step}")

    # validate(val_loader, model, lpips, start_epoch - 1, local_rank)

    # train, validate
    for epoch in range(start_epoch, epochs):
        # resume from epoch and step
        train_sampler.set_epoch(epoch)
        train_loader.set_step(start_step)

        train(train_loader, model, lpips, opt_g, opt_d, epoch, local_rank, start_step, best_score)
        start_step = 0  # reset

        loss_recon = validate(val_loader, model, lpips, epoch, local_rank)

        # save
        if rank == 0:
            state = {
                "model": model.module.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch + 1,
                "step": 0,
                "loss_recon": min(loss_recon, best_score),
            }
            save_model(state, save_path / "latest.pt")

            if epoch % 20 == 0 and epoch == epochs - 1:
                save_model(state, save_path / f"{epoch:04d}.pt")

            if loss_recon < best_score:
                best_score = loss_recon
                save_model(state, save_path / "best.pt")
                print(f"New Best loss_recon: {loss_recon:.3f}", flush=True)

    if writer:
        writer.close()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
