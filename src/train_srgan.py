"""Training script for SRGAN (SRResNet generator + discriminator)."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import yaml

from src.config import load_config
from src.datasets.div2k import make_div2k_loaders
from src.models.factory import create_model
from src.models.srgan_discriminator import SRGANDiscriminator
from src.utils.device import get_device
from src.utils.losses import PerceptualLoss, GANLoss
from src.utils.metrics import psnr, ssim
from src.utils.seed import set_seed


def tensor_to_image(tensor):
    """Convert tensor (C, H, W) in [0, 1] to PIL Image."""
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = (tensor * 255).byte()
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def build_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    cfg,
    epoch,
    g_loss,
    d_loss,
    psnr_value,
    ssim_value,
    best_psnr,
    run_dir,
    global_step,
    phase,
    scaler_g=None,
    scaler_d=None,
):
    checkpoint = {
        "model": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "g_loss": g_loss,
        "d_loss": d_loss,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "best_psnr": best_psnr,
        "run_dir": str(run_dir),
        "global_step": global_step,
        "phase": phase,
    }
    if scaler_g is not None:
        checkpoint["scaler_g"] = scaler_g.state_dict()
    if scaler_d is not None:
        checkpoint["scaler_d"] = scaler_d.state_dict()
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train SRGAN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"].lower()
    if model_name != "srgan":
        raise ValueError(f"Config model.name must be 'srgan' for this script, got '{model_name}'")

    set_seed(cfg["project"]["seed"])
    device = get_device()
    train_loader, val_loader = make_div2k_loaders(cfg)

    scale = cfg["data"]["scale"]
    epochs = int(cfg["train"]["epochs"])
    base_lr = float(cfg["train"]["lr"])
    lr_g = float(cfg["train"].get("lr_g", base_lr))
    lr_d = float(cfg["train"].get("lr_d", base_lr))
    betas = cfg["train"].get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("train.betas must be a list/tuple with two values, e.g. [0.9, 0.999]")
    betas = (float(betas[0]), float(betas[1]))

    save_every = int(cfg["train"].get("save_every", 1))
    val_every = int(cfg["train"].get("val_every", 1))
    output_root = Path(cfg["paths"]["output_root"])

    srgan_cfg = cfg["train"].get("srgan", {})
    pretrain_epochs = int(srgan_cfg.get("pretrain_epochs", 10))
    lambda_pixel = float(srgan_cfg.get("lambda_pixel", 1.0))
    lambda_perceptual = float(srgan_cfg.get("lambda_perceptual", 1.0))
    lambda_gan = float(srgan_cfg.get("lambda_gan", 1e-3))
    d_updates_per_g = int(srgan_cfg.get("d_updates_per_g", 1))
    label_smoothing = float(srgan_cfg.get("label_smoothing", 0.0))
    vgg_feature_layer = int(srgan_cfg.get("vgg_feature_layer", 35))
    vgg_pretrained = bool(srgan_cfg.get("vgg_pretrained", True))

    if pretrain_epochs < 0:
        raise ValueError("train.srgan.pretrain_epochs must be >= 0")
    if d_updates_per_g < 1:
        raise ValueError("train.srgan.d_updates_per_g must be >= 1")

    disc_cfg = cfg.get("model", {}).get("discriminator", {})
    disc_base_channels = int(disc_cfg.get("base_channels", 64))

    generator = create_model("srgan", cfg, device)
    discriminator = SRGANDiscriminator(base_channels=disc_base_channels).to(device)

    pixel_criterion = nn.L1Loss()
    perceptual_criterion = PerceptualLoss(
        feature_layer=vgg_feature_layer, pretrained=vgg_pretrained
    ).to(device)
    gan_criterion = GANLoss(label_smoothing=label_smoothing).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

    use_amp = bool(cfg["runtime"]["amp"]) and device.type == "cuda"
    scaler_g = torch.cuda.amp.GradScaler() if use_amp else None
    scaler_d = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    global_step_offset = 0
    best_psnr = -float("inf")
    run_dir = None

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        generator.load_state_dict(checkpoint["model"])
        if "discriminator" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator"])

        if "optimizer_g" in checkpoint:
            optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            print("Loaded generator optimizer state")
        if "optimizer_d" in checkpoint:
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            print("Loaded discriminator optimizer state")

        if use_amp and scaler_g is not None and "scaler_g" in checkpoint:
            scaler_g.load_state_dict(checkpoint["scaler_g"])
        if use_amp and scaler_d is not None and "scaler_d" in checkpoint:
            scaler_d.load_state_dict(checkpoint["scaler_d"])

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            global_step_offset = checkpoint.get("global_step", start_epoch * len(train_loader))
            print(f"Resuming from epoch {start_epoch}")

        if "best_psnr" in checkpoint:
            best_psnr = checkpoint["best_psnr"]
            print(f"Resuming best PSNR: {best_psnr:.4f} dB")

        if "run_dir" in checkpoint:
            run_dir = Path(checkpoint["run_dir"])
            print(f"Continuing in existing run directory: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Checkpoint missing 'run_dir'. Cannot resume.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_root / f"srgan_x{scale}" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    writer = SummaryWriter(log_dir=str(run_dir))

    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Model: SRGAN, Scale: x{scale}")
    print(f"Epochs: {epochs}, AMP: {use_amp}")
    print(f"LR (G/D): {lr_g} / {lr_d}, Betas: {betas}")
    print(
        "Loss weights "
        f"(pixel/perceptual/gan): {lambda_pixel} / {lambda_perceptual} / {lambda_gan}"
    )
    print(f"Pretrain epochs: {pretrain_epochs}, D updates per G: {d_updates_per_g}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print()

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_offset = epoch - start_epoch
        current_global_step = global_step_offset + (epoch_offset + 1) * len(train_loader)

        pretrain_phase = epoch < pretrain_epochs
        phase_name = "pretrain" if pretrain_phase else "adversarial"

        generator.train()
        if pretrain_phase:
            discriminator.eval()
        else:
            discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_pixel_loss = 0.0
        total_perceptual_loss = 0.0
        total_gan_loss = 0.0
        num_batches = 0

        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            global_step = global_step_offset + epoch_offset * len(train_loader) + batch_idx

            if pretrain_phase:
                optimizer_g.zero_grad()
                if use_amp:
                    with torch.autocast(device_type="cuda", enabled=True):
                        pred_batch = generator(lr_batch)
                        pixel_loss = pixel_criterion(pred_batch, hr_batch)
                        g_loss = lambda_pixel * pixel_loss
                    scaler_g.scale(g_loss).backward()
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    pred_batch = generator(lr_batch)
                    pixel_loss = pixel_criterion(pred_batch, hr_batch)
                    g_loss = lambda_pixel * pixel_loss
                    g_loss.backward()
                    optimizer_g.step()

                d_loss_value = 0.0
                perceptual_loss_value = 0.0
                gan_loss_value = 0.0

            else:
                d_loss_accum = 0.0
                for _ in range(d_updates_per_g):
                    optimizer_d.zero_grad()
                    if use_amp:
                        with torch.autocast(device_type="cuda", enabled=True):
                            with torch.no_grad():
                                fake_detached = generator(lr_batch).detach()
                            pred_real = discriminator(hr_batch)
                            pred_fake = discriminator(fake_detached)
                            d_loss = gan_criterion.discriminator_loss(pred_real, pred_fake)
                        scaler_d.scale(d_loss).backward()
                        scaler_d.step(optimizer_d)
                        scaler_d.update()
                    else:
                        with torch.no_grad():
                            fake_detached = generator(lr_batch).detach()
                        pred_real = discriminator(hr_batch)
                        pred_fake = discriminator(fake_detached)
                        d_loss = gan_criterion.discriminator_loss(pred_real, pred_fake)
                        d_loss.backward()
                        optimizer_d.step()
                    d_loss_accum += d_loss.item()

                d_loss_value = d_loss_accum / max(1, d_updates_per_g)

                optimizer_g.zero_grad()
                if use_amp:
                    with torch.autocast(device_type="cuda", enabled=True):
                        pred_batch = generator(lr_batch)
                        pixel_loss = pixel_criterion(pred_batch, hr_batch)
                        perceptual_loss = perceptual_criterion(pred_batch, hr_batch)
                        pred_fake_for_g = discriminator(pred_batch)
                        gan_loss = gan_criterion.generator_loss(pred_fake_for_g)
                        g_loss = (
                            lambda_pixel * pixel_loss
                            + lambda_perceptual * perceptual_loss
                            + lambda_gan * gan_loss
                        )
                    scaler_g.scale(g_loss).backward()
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    pred_batch = generator(lr_batch)
                    pixel_loss = pixel_criterion(pred_batch, hr_batch)
                    perceptual_loss = perceptual_criterion(pred_batch, hr_batch)
                    pred_fake_for_g = discriminator(pred_batch)
                    gan_loss = gan_criterion.generator_loss(pred_fake_for_g)
                    g_loss = (
                        lambda_pixel * pixel_loss
                        + lambda_perceptual * perceptual_loss
                        + lambda_gan * gan_loss
                    )
                    g_loss.backward()
                    optimizer_g.step()

                perceptual_loss_value = perceptual_loss.item()
                gan_loss_value = gan_loss.item()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss_value
            total_pixel_loss += pixel_loss.item()
            total_perceptual_loss += perceptual_loss_value
            total_gan_loss += gan_loss_value
            num_batches += 1

            writer.add_scalar("train/g_loss_step", g_loss.item(), global_step)
            writer.add_scalar("train/d_loss_step", d_loss_value, global_step)
            writer.add_scalar("train/pixel_loss_step", pixel_loss.item(), global_step)
            writer.add_scalar("train/perceptual_loss_step", perceptual_loss_value, global_step)
            writer.add_scalar("train/gan_loss_step", gan_loss_value, global_step)

        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        avg_pixel_loss = total_pixel_loss / num_batches
        avg_perceptual_loss = total_perceptual_loss / num_batches
        avg_gan_loss = total_gan_loss / num_batches

        writer.add_scalar("train/g_loss_epoch", avg_g_loss, epoch)
        writer.add_scalar("train/d_loss_epoch", avg_d_loss, epoch)
        writer.add_scalar("train/pixel_loss_epoch", avg_pixel_loss, epoch)
        writer.add_scalar("train/perceptual_loss_epoch", avg_perceptual_loss, epoch)
        writer.add_scalar("train/gan_loss_epoch", avg_gan_loss, epoch)
        writer.add_scalar("train/phase_adversarial", 0 if pretrain_phase else 1, epoch)

        avg_psnr = None
        avg_ssim = None
        if (epoch + 1) % val_every == 0:
            generator.eval()
            total_psnr = 0.0
            total_ssim = 0.0
            val_num_samples = 0
            with torch.no_grad():
                for lr_val, hr_val in val_loader:
                    lr_val = lr_val.to(device)
                    hr_val = hr_val.to(device)

                    if use_amp:
                        with torch.autocast(device_type="cuda", enabled=True):
                            pred_val = generator(lr_val)
                    else:
                        pred_val = generator(lr_val)

                    batch_psnr = psnr(pred_val, hr_val)
                    batch_ssim = ssim(pred_val, hr_val)

                    batch_size = lr_val.shape[0]
                    total_psnr += batch_psnr * batch_size
                    total_ssim += batch_ssim * batch_size
                    val_num_samples += batch_size

            avg_psnr = total_psnr / val_num_samples
            avg_ssim = total_ssim / val_num_samples
            writer.add_scalar("val/psnr", avg_psnr, epoch)
            writer.add_scalar("val/ssim", avg_ssim, epoch)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_checkpoint = build_checkpoint(
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    cfg=cfg,
                    epoch=epoch,
                    g_loss=avg_g_loss,
                    d_loss=avg_d_loss,
                    psnr_value=avg_psnr,
                    ssim_value=avg_ssim,
                    best_psnr=best_psnr,
                    run_dir=run_dir,
                    global_step=current_global_step,
                    phase=phase_name,
                    scaler_g=scaler_g if use_amp else None,
                    scaler_d=scaler_d if use_amp else None,
                )
                torch.save(best_checkpoint, run_dir / "best.pth")
                print(f"  New best PSNR: {best_psnr:.4f} dB (saved best.pth)")

        if (epoch + 1) % save_every == 0:
            generator.eval()
            with torch.no_grad():
                lr_example, hr_example = next(iter(val_loader))
                lr_example = lr_example.to(device)
                hr_example = hr_example.to(device)

                if use_amp:
                    with torch.autocast(device_type="cuda", enabled=True):
                        pred_example = generator(lr_example)
                else:
                    pred_example = generator(lr_example)

                pred_example = pred_example.clamp(0.0, 1.0)

                examples_dir = run_dir / "examples" / f"ep{epoch}"
                examples_dir.mkdir(parents=True, exist_ok=True)

                num_examples = min(4, lr_example.shape[0])
                for i in range(num_examples):
                    sample_dir = examples_dir / f"sample_{i}"
                    sample_dir.mkdir(exist_ok=True)

                    lr_bicubic = F.interpolate(
                        lr_example[i : i + 1],
                        size=(hr_example.shape[2], hr_example.shape[3]),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(0)

                    tensor_to_image(lr_bicubic).save(sample_dir / "bicubic.png")
                    tensor_to_image(pred_example[i]).save(sample_dir / "pred.png")
                    tensor_to_image(hr_example[i]).save(sample_dir / "hr.png")

        last_checkpoint = build_checkpoint(
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            cfg=cfg,
            epoch=epoch,
            g_loss=avg_g_loss,
            d_loss=avg_d_loss,
            psnr_value=avg_psnr,
            ssim_value=avg_ssim,
            best_psnr=best_psnr,
            run_dir=run_dir,
            global_step=current_global_step,
            phase=phase_name,
            scaler_g=scaler_g if use_amp else None,
            scaler_d=scaler_d if use_amp else None,
        )
        torch.save(last_checkpoint, run_dir / "last.pth")
        torch.save(last_checkpoint, run_dir / f"ckpt_ep{epoch}.pth")

        epoch_time = time.time() - epoch_start_time
        psnr_str = f"{avg_psnr:.4f} dB" if avg_psnr is not None else "N/A"
        ssim_str = f"{avg_ssim:.6f}" if avg_ssim is not None else "N/A"
        print(
            f"Epoch {epoch + 1}/{epochs} [{phase_name}], "
            f"G loss: {avg_g_loss:.6f}, D loss: {avg_d_loss:.6f}, "
            f"PSNR: {psnr_str}, SSIM: {ssim_str}, Time: {epoch_time:.1f}s"
        )

    writer.close()
    print(f"\nTraining completed. Checkpoints saved to: {run_dir}")
    print(f"Best PSNR: {best_psnr:.4f} dB")


if __name__ == "__main__":
    main()
