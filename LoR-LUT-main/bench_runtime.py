import argparse
import time
import contextlib

import torch
import torch.nn.functional as F

from core.core_lut import LoRIA3DLUT, cp_residual_to_lut


def _sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms():
    return time.perf_counter() * 1000.0


@torch.no_grad()
def bench_once(model: LoRIA3DLUT, device: str, H: int = 2160, W: int = 3840, amp: bool = False):
    model.eval()

    # Inputs
    x = torch.rand(1, 3, H, W, device=device)
    x_lr = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

    # Autocast only on CUDA when requested
    if device.startswith("cuda") and torch.cuda.is_available():
        amp_ctx = torch.amp.autocast("cuda", enabled=amp)
    else:
        amp_ctx = contextlib.nullcontext()

    with amp_ctx:
        _sync(device); t0 = _ms()
        # Weight predictor
        alpha = model.weight_pred(x_lr)
        _sync(device); t1 = _ms()

        # Residual (may be disabled when R <= 0)
        if getattr(model, "R", 0) > 0 and getattr(model, "resid_pred", None) is not None:
            u, v, w, c = model.resid_pred(x_lr)
            _sync(device); t2 = _ms()
            delta = cp_residual_to_lut(u, v, w, c)
            _sync(device); t3 = _ms()
        else:
            delta = torch.zeros(1, model.G, model.G, model.G, 3, device=device, dtype=model.bases.dtype)
            t2 = t1
            t3 = t1

        # Fuse and apply
        L = model.fuse_bases(alpha) + delta
        _sync(device); t4 = _ms()
        out = model.apply_lut(x, L)
        _sync(device); t5 = _ms()

    return {
        "weight_pred_ms": t1 - t0,
        "resid_pred_ms": t2 - t1,
        "cp_residual_ms": t3 - t2,
        "fuse_ms": t4 - t3,
        "apply_lut_ms": t5 - t4,
        "total_ms": t5 - t0,
    }


def bench(model: LoRIA3DLUT, device: str, warmup: int = 20, iters: int = 50, amp: bool = False):
    # Warmup
    for _ in range(warmup):
        bench_once(model, device, amp=amp)

    # Measure
    acc = None
    for _ in range(iters):
        t = bench_once(model, device, amp=amp)
        if acc is None:
            acc = {k: 0.0 for k in t}
        for k, v in t.items():
            acc[k] += v
    for k in acc:
        acc[k] /= iters
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for benchmarking")
    parser.add_argument("--G", type=int, default=33)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--from_ckpt", type=str, default="", help="Path to checkpoint to load")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--H", type=int, default=2160, help="Image height (default 4K UHD)")
    parser.add_argument("--W", type=int, default=3840, help="Image width (default 4K UHD)")
    args = parser.parse_args()

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Device: {device} | AMP: {args.amp}")

    if args.from_ckpt:
        ckpt = torch.load(args.from_ckpt, map_location=device)
        cfg = ckpt["cfg"]
        G = cfg["model"]["G"]; K = cfg["model"]["K"]; R = cfg["model"]["R"]
        model = LoRIA3DLUT(G=G, K=K, R=R).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"Loaded checkpoint with G={G}, K={K}, R={R}")
    else:
        model = LoRIA3DLUT(G=args.G, K=args.K, R=args.R).to(device)
        print(f"Fresh model G={args.G}, K={args.K}, R={args.R}")

    torch.set_grad_enabled(False)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Warmup
    for _ in range(args.warmup):
        bench_once(model, device, H=args.H, W=args.W, amp=args.amp)

    # Measure
    acc = None
    for _ in range(args.iters):
        t = bench_once(model, device, H=args.H, W=args.W, amp=args.amp)
        if acc is None:
            acc = {k: 0.0 for k in t}
        for k, v in t.items():
            acc[k] += v
    for k in acc:
        acc[k] /= args.iters

    print("--- Averages (ms) ---")
    for k in ["weight_pred_ms", "resid_pred_ms", "cp_residual_ms", "fuse_ms", "apply_lut_ms", "total_ms"]:
        print(f"{k}: {acc[k]:.3f} ms")

    mpix = (args.H * args.W) / 1e6
    total = acc["total_ms"]
    throughput = mpix / (total / 1000.0)
    print(f"{args.H}x{args.W} latency: {total:.3f} ms | Throughput: {throughput:.1f} MP/s")


if __name__ == "__main__":
    main()

