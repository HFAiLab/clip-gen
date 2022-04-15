import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from models.vqgan import VQGAN
from models.clip import clip_vit_b32
from models.gpt import __dict__ as models
from datasets.statistic import mean, std
from tokenizer import tokenize
from utils import CLIPWrapper


DEFAULT_TEXT = 'A photo of a tower in front of a mountain'
# 'A photo of a living area with a television and table'
# 'A city bus driving on the city street'
# 'A train being operated on a train track'
# 'The reflection of the house in the water'
# 'A woman is skiing on a white mountain'

parser = argparse.ArgumentParser(description='CLIP-GEN demo')
parser.add_argument('--text', type=str, default=DEFAULT_TEXT, help='input text')
parser.add_argument('--out', type=str, default='sample.jpg', help='output image path')
parser.add_argument('--cand-size', type=int, default=64, help='number of candidate images')
parser.add_argument('--out-k', type=int, default=8, help='number of sample images to be saved')
args = parser.parse_args()


torch.set_grad_enabled(False)
device = torch.device('cuda', 0)

gpt_name = "gpt2_medium"
dataset_name = "coco"

codebook_size = 16384
embed_dim = 256
dropout = 0.1
normalize_clip = True

batch_size = 8
vqgan_ckpt = f"pretrained/vqgan_{dataset_name}.pt"
gpt_ckpt = f"pretrained/gpt_{dataset_name}.pt"

text = args.text
candidate_size = args.cand_size
out_k = args.out_k
top_k = 500
top_p = 0.95
bs = 8  # batch size
assert candidate_size % bs == 0

##################################
# VQGAN
##################################
vqgan = VQGAN(codebook_size, embed_dim).to(device).eval()
state = torch.load(vqgan_ckpt, map_location='cpu')
vqgan.load_state_dict(state['model'])
print(f"Loaded VQGAN model from {vqgan_ckpt}, epoch {state['epoch']}")

##################################
# GPT
##################################
gpt = models[gpt_name](vocab_size=codebook_size, dropout=dropout).to(device).eval()
state = torch.load(gpt_ckpt, map_location='cpu')
gpt.load_state_dict(state['model'])
print(f"Loaded GPT model from {gpt_ckpt}, epoch {state['epoch']}")

##################################
# CLIP
##################################
clip = clip_vit_b32(pretrained=True).to(device).eval()
clip = CLIPWrapper(clip, normalize=normalize_clip)


##################################
# sample
##################################
print("Input text:", text)
texts = [text]
texts = tokenize(texts).to(device)

x_recons = []

text_embeddings = clip.encode_text(texts) # [1, 512]
embeds = text_embeddings.expand(bs, -1)
for i in range(candidate_size // bs):
    z_idx = gpt.sample(embeds, steps=16 * 16, top_k=top_k, top_p=top_p)  # [-1, 16*16]
    z_idx = z_idx.view(-1, 16, 16)
    z = vqgan.quantizer.decode(z_idx)  # (B, H, W, C)
    z = z.permute(0, 3, 1, 2)  # [B, C, H, W]
    x_recon = vqgan.decode(z)  # [B, 3, H, W]
    x_recons.append(x_recon)

# torch.cuda.empty_cache()
x_recon = torch.cat(x_recons, dim=0)


##################################
# filter by CLIP
##################################

clip_x_recon = F.interpolate(x_recon, 224, mode='bilinear')

img_embeddings = []
for i in range(candidate_size // bs):
    embd = clip.encode_image(clip_x_recon[i * bs:(i+1) * bs])  # [B, 512]
    img_embeddings.append(embd)
    torch.cuda.empty_cache()
img_embeddings = torch.cat(img_embeddings, dim=0)

sim = F.cosine_similarity(text_embeddings, img_embeddings)
topk = sim.argsort(descending=True)[:out_k]
print("CLIP similarity", sim[topk])


##################################
# display image
##################################

x = x_recon[topk]
std = torch.tensor(std).view(1, -1, 1, 1).to(x)
mean = torch.tensor(mean).view(1, -1, 1, 1).to(x)
img = x.clone()  # [2 * N, 3, H, W]
img = img * std + mean
img = make_grid(img, nrow=min(x.size(0), 4))
img = img.permute(1, 2, 0).clamp(0, 1)

plt.imshow(img.cpu())
plt.title(text)
plt.axis('off')
plt.savefig(args.out, bbox_inches='tight')
