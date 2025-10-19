# ddpm_train_fixed_amp.py
import os
import math
from datetime import datetime
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

# ---------- User config (kept from your original) ----------
data_root = "/mnt/d/data/face/img/img_align_celeba"
save_dir = "./ddpm_checkpoints"
os.makedirs(save_dir, exist_ok=True)
batch_size = 8
lr = 1e-5
num_epochs = 100
image_size = 104
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4
pin_memory = True
sample_every = 1
num_sample_images = 8
base_ch = 128
T = 1000  # diffusion timesteps

# Use AMP if CUDA is available
use_amp = torch.cuda.is_available() and hasattr(torch.cuda, "amp")

# ---------- Data ----------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # [0,1]
])


class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # scale to [-1,1]
        img = img * 2.0 - 1.0
        return img


dataset = CelebADataset(root=data_root, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory)


# ---------- Diffusion schedule utilities ----------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(T).to(device)  # shape [T]
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)  # \bar{\alpha}_t
alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alpha_cumprod[:-1]], dim=0)
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

# precompute terms for sampling
posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)


# ---------- Time embedding ----------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) longs
        device = t.device
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


# ---------- Enhanced UNet with Residual Blocks + Attention ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)
        h = self.conv1(x)
        t_emb = self.time_emb_proj(t_emb)
        h = h + t_emb[:, :, None, None]
        h = self.conv2(h)
        return h + residual


# ---------- Self-Attention Layer ----------
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        # reshape to (B, heads, C//heads, H*W)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return x + out  # residual connection


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, t_emb):
        skips = []
        for block in self.blocks:
            x = block(x, t_emb)
            skips.append(x)
        x = self.attn(x)
        x = self.downsample(x)
        return x, skips


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, upsample=True, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skips, t_emb):
        x = self.upsample(x)
        for block in self.blocks:
            if skips:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
        x = self.attn(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(channels)  # 加注意力层

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.attn(x)
        return x


# ---------- Full Enhanced UNet with Attention ----------
class EnhancedUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # Down blocks (添加注意力在 deeper layers)
        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, downsample=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, num_res_blocks)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, num_res_blocks)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, num_res_blocks, use_attention=True)

        # Middle block
        self.mid = MidBlock(base_ch * 8, time_emb_dim, num_res_blocks * 2)

        # Up blocks (同样添加注意力)
        self.up4 = UpBlock(base_ch * 8, base_ch * 4, time_emb_dim, num_res_blocks, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, time_emb_dim, num_res_blocks)
        self.up2 = UpBlock(base_ch * 2, base_ch, time_emb_dim, num_res_blocks)
        self.up1 = UpBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, upsample=False)

        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        x, s1 = self.down1(x, t_emb); skips.extend(s1)
        x, s2 = self.down2(x, t_emb); skips.extend(s2)
        x, s3 = self.down3(x, t_emb); skips.extend(s3)
        x, s4 = self.down4(x, t_emb); skips.extend(s4)

        x = self.mid(x, t_emb)

        x = self.up4(x, skips, t_emb)
        x = self.up3(x, skips, t_emb)
        x = self.up2(x, skips, t_emb)
        x = self.up1(x, skips, t_emb)

        return self.final(x)


# ---------- Diffusion forward q_sample ----------
def q_sample(x_start, t, noise=None):
    """
    x_start: (B,C,H,W) in [-1,1]
    t: tensor of shape (B,) with values in [0,T-1]
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_cumprod_t = sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise, noise


# ---------- Loss (predict noise) ----------
def p_losses(model, x_start, t):
    x_noisy, noise = q_sample(x_start, t)
    predicted_noise = model(x_noisy, t)
    loss = F.mse_loss(predicted_noise, noise, reduction='mean')
    return loss


# ---------- Sampling (ancestral sampling from DDPM) ----------
@torch.no_grad()
def p_sample(model, x_t, t):
    """
    one reverse step from x_t at timestep t -> x_{t-1}
    """
    # 当前步的参数
    alpha_t = alphas[t]
    alpha_cumprod_t = alpha_cumprod[t]
    alpha_cumprod_prev_t = alpha_cumprod_prev[t]

    # 预测噪声 ε
    pred_noise = model(x_t, torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device))

    # ---- 计算均值 μ ----
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

    mu = (1.0 / sqrt_alpha_t) * (
        x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred_noise
    )

    # ---- 计算方差 σ² ----
    sigma2 = ((1.0 - alpha_t) * (1.0 - alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t)
    sigma = torch.sqrt(sigma2)

    # ---- 采样 x_{t-1} ----
    if t == 0:
        return mu
    else:
        noise = torch.randn_like(x_t)
        return mu + sigma.view(1, 1, 1, 1) * noise



@torch.no_grad()
def sample_loop(model, batch_size, device):
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for t in reversed(range(T)):
        # use autocast during sampling for better performance on GPU
        if use_amp:
            with torch.cuda.amp.autocast():
                x = p_sample(model, x, t)
        else:
            x = p_sample(model, x, t)
    # x in [-1,1], convert to [0,1]
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    return x


# ---------- Utilities ----------
def save_checkpoint(model, optim, epoch, path, scaler=None):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict()
    }
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    torch.save(state, path)


def save_image_grid(tensor, filename, nrow=8):
    # tensor expected in [0,1]
    utils.save_image(tensor, filename, nrow=nrow, padding=2)


# ---------- Training ----------
def train():
    model = EnhancedUNet(in_ch=3, base_ch=base_ch, time_emb_dim=512, num_res_blocks=2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1

    # optionally load last checkpoint if present (and scaler state)
    last_ckpt = os.path.join(save_dir, "ddpm_last.pth")
    if os.path.exists(last_ckpt):
        print("Loading checkpoint:", last_ckpt)
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model_state"])
        optim.load_state_dict(ck["optim_state"])
        if "scaler_state" in ck and use_amp:
            try:
                scaler.load_state_dict(ck["scaler_state"])
            except Exception:
                print("Warning: failed to load scaler state (version mismatch?)")
        start_epoch = ck["epoch"] + 1
        print("Resumed from epoch", start_epoch)

    global_step = 0
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device)  # in [-1,1]
            bs = imgs.size(0)

            # sample t for each image in the batch (uniform 0..T-1)
            t = torch.randint(0, T, (bs,), device=device).long()

            optim.zero_grad()

            # forward + loss within autocast if AMP enabled
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = p_losses(model, imgs, t)
                # scale -> backward
                scaler.scale(loss).backward()
                # unscale before clipping
                scaler.unscale_(optim)
                # Gradient clipping for stability (unscaled grads)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss = p_losses(model, imgs, t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if batch_idx % 50 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Epoch {epoch}/{num_epochs} Batch {batch_idx}/{len(loader)} "
                      f"Loss {loss.item():.6f}")

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"=== Epoch {epoch} finished. Avg loss: {avg_loss:.6f} ===")

        # sample images every few epochs
        if epoch % sample_every == 0 or epoch == 1:
            # save checkpoint (include scaler state if using AMP)
            ckpt_path = os.path.join(save_dir, f"ddpm_epoch{epoch}.pth")
            save_checkpoint(model, optim, epoch, ckpt_path, scaler if use_amp else None)
            # update last
            save_checkpoint(model, optim, epoch, last_ckpt, scaler if use_amp else None)

            model.eval()
            with torch.no_grad():
                samples = sample_loop(model, num_sample_images, device)
            # save grid
            save_image_grid(samples, os.path.join(save_dir, f"sample_epoch{epoch}.png"), nrow=8)
            model.train()

    print("Training complete.")


def merge_models(model_checkpoints, output_path, merge_method='average'):
    """
    合并多个模型检查点

    Args:
        model_checkpoints: 模型检查点路径列表
        output_path: 合并后模型的保存路径
        merge_method: 合并方法，'average'为平均权重，'ema'为指数移动平均
    """
    print(f"开始合并 {len(model_checkpoints)} 个模型检查点...")

    # 加载第一个模型作为基础
    base_checkpoint = torch.load(model_checkpoints[0], map_location='cpu')
    merged_state_dict = base_checkpoint['model_state'].copy()

    if merge_method == 'average':
        # 平均权重
        for checkpoint_path in model_checkpoints[1:]:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            for key in merged_state_dict.keys():
                merged_state_dict[key] += checkpoint['model_state'][key]

        for key in merged_state_dict.keys():
            merged_state_dict[key] = merged_state_dict[key] / len(model_checkpoints)

    elif merge_method == 'ema':
        # 指数移动平均 (EMA)，越新的模型权重越大
        alpha = 0.9  # EMA衰减因子
        weight = 1.0

        for i, checkpoint_path in enumerate(model_checkpoints[1:], 1):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            current_weight = weight * (alpha ** (len(model_checkpoints) - i - 1))

            for key in merged_state_dict.keys():
                merged_state_dict[key] = (merged_state_dict[key] * (1 - current_weight) +
                                          checkpoint['model_state'][key] * current_weight)

    # 保存合并后的模型
    merged_checkpoint = {
        'model_state': merged_state_dict,
        'epoch': f"merged_from_{len(model_checkpoints)}_models",
        'merge_method': merge_method
    }

    torch.save(merged_checkpoint, output_path)
    print(f"合并完成！模型已保存至: {output_path}")

    return merged_state_dict


def generate_images_with_merged_model(model_path, num_images=16, output_dir="./merged_model_samples"):
    """使用合并后的模型生成图片"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载合并模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 创建模型并加载权重
    model = EnhancedUNet(in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print("开始生成图片...")
    with torch.no_grad():
        samples = sample_loop(model, num_images, device)

    # 保存生成的图片
    output_filename = os.path.join(output_dir, f"merged_model_samples.png")
    save_image_grid(samples, output_filename, nrow=4)
    print(f"图片已保存至: {output_filename}")

    # 同时保存一些中间结果（可选）
    print("生成多组图片...")
    for i in range(3):
        with torch.no_grad():
            samples = sample_loop(model, num_images, device)
        output_filename = os.path.join(output_dir, f"merged_model_samples_set_{i + 1}.png")
        save_image_grid(samples, output_filename, nrow=4)
        print(f"第 {i + 1} 组图片已保存")


if __name__ == "__main__":
    print("Starting DDPM training on device:", device)
    print("AMP enabled:", use_amp)
    print("Dataset size:", len(dataset))
    train()
