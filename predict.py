import tempfile
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from cog import BasePredictor, Path, Input

from models.vqgan import VQGAN
from models.clip import clip_vit_b32
from models.gpt import __dict__ as models
from tokenizer import tokenize
from utils import CLIPWrapper


class Predictor(BasePredictor):
    def setup(self):

        torch.set_grad_enabled(False)
        self.device = torch.device('cuda', 0)
        gpt_name = "gpt2_medium"
        dataset_name = "coco"

        codebook_size = 16384
        embed_dim = 256
        dropout = 0.1
        normalize_clip = True

        batch_size = 8
        vqgan_ckpt = f"pretrained/vqgan_{dataset_name}.pt"
        gpt_ckpt = f"pretrained/gpt_{dataset_name}.pt"

        ##################################
        # VQGAN
        ##################################
        self.vqgan = VQGAN(codebook_size, embed_dim).to(self.device).eval()
        state = torch.load(vqgan_ckpt, map_location='cpu')
        self.vqgan.load_state_dict(state['model'])
        print(f"Loaded VQGAN model from {vqgan_ckpt}, epoch {state['epoch']}")

        ##################################
        # GPT
        ##################################
        self.gpt = models[gpt_name](vocab_size=codebook_size, dropout=dropout).to(self.device).eval()
        state = torch.load(gpt_ckpt, map_location='cpu')
        self.gpt.load_state_dict(state['model'])
        print(f"Loaded GPT model from {gpt_ckpt}, epoch {state['epoch']}")

        ##################################
        # CLIP
        ##################################
        self.clip = clip_vit_b32(pretrained=True).to(self.device).eval()
        self.clip = CLIPWrapper(self.clip, normalize=normalize_clip)

    def predict(
        self,
        text: str = Input(
            description="Text for generating image.",
            default="A train being operated on a train track"
        ),
        num_samples: int = Input(
            description="Number of samples to generate.",
            default=8,
            ge=1,
            le=8
        ),
    ) -> Path:

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        candidate_size = 32
        top_k = 500
        top_p = 0.95
        bs = 8  # batch size
        assert candidate_size % bs == 0

        texts = [text]
        texts = tokenize(texts).to(self.device)

        x_recons = []

        text_embeddings = self.clip.encode_text(texts)  # [1, 512]
        embeds = text_embeddings.expand(bs, -1)
        for i in range(candidate_size // bs):
            z_idx = self.gpt.sample(embeds, steps=16 * 16, top_k=top_k, top_p=top_p)  # [-1, 16*16]
            z_idx = z_idx.view(-1, 16, 16)
            z = self.vqgan.quantizer.decode(z_idx)  # (B, H, W, C)
            z = z.permute(0, 3, 1, 2)  # [B, C, H, W]
            x_recon = self.vqgan.decode(z)  # [B, 3, H, W]
            x_recons.append(x_recon)

        # torch.cuda.empty_cache()
        x_recon = torch.cat(x_recons, dim=0)

        ##################################
        # filter by CLIP
        ##################################

        clip_x_recon = F.interpolate(x_recon, 224, mode='bilinear')

        img_embeddings = []

        for i in range(candidate_size // bs):
            embd = self.clip.encode_image(clip_x_recon[i * bs:(i + 1) * bs])  # [B, 512]
            img_embeddings.append(embd)
            torch.cuda.empty_cache()
        img_embeddings = torch.cat(img_embeddings, dim=0)

        sim = F.cosine_similarity(text_embeddings, img_embeddings)
        topk = sim.argsort(descending=True)[:num_samples]
        print("CLIP similarity", sim[topk])

        ##################################
        # save image
        ##################################

        out_path = Path(tempfile.mkdtemp()) / "output.png"
        x = x_recon[topk]
        std = torch.tensor(std).view(1, -1, 1, 1).to(x)
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(x)
        img = x.clone()  # [2 * N, 3, H, W]
        img = img * std + mean
        print(img)
        print(type(img))
        img = make_grid(img, nrow=min(x.size(0), 4))
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.imshow(img.cpu())
        plt.title(text)
        plt.axis('off')
        plt.savefig(str(out_path), bbox_inches='tight')
        return out_path
