import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from model import *

def output_compare(dataset, models, outputs, idx):
    img_noi = dataset[idx][0].cpu().numpy()
    img_noi = np.transpose(img_noi, (1, 2, 0))
    img_gt = dataset[idx][1].cpu().numpy()
    img_gt = np.transpose(img_gt, (1, 2, 0))

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(2, len(models), 1)
    plt.imshow(img_gt)
    plt.title('Ground truth')
    plt.subplot(2, len(models), 2)
    plt.imshow(img_noi)
    plt.title('Noisy')

    for i, model in enumerate(models):
        plt.subplot(2, len(models), len(models)+1+i)
        plt.imshow(outputs[idx])
        plt.title(model.name)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.savefig('output_comparision.png')

    # plt.imsave('test_gt.png', arr=img_gt)
    # plt.imsave('test_u_net.png', arr=img_u_net)

def _logmag(x, eps=1e-8):
    # x: torch tensor (N,C,H,W), values in [0,1]
    F = torch.fft.fft2(x, dim=(-2, -1))        # full complex spectrum
    mag = F.abs().clamp_min(eps)               # avoid log(0)
    return torch.log(mag)                      # log-magnitude

def _freq_error_map(target, pred, eps=1e-8):
    """
    pred/target: torch tensor (N,C,H,W) in [0,1]
    returns: torch tensor (N,H,W) log-mag MSE per pixel in frequency plane
    """
    Lt = _logmag(target, eps)
    Lp = _logmag(pred, eps)    # (N,C,H,W)
    err = (Lp - Lt)**2         # (N,C,H,W)
    err = err.mean(dim=1)      # average over channels -> (N,H,W)
    # center DC for visualization
    err = torch.fft.fftshift(err, dim=(-2, -1))
    return err

def get_error_per_radius(img2d):
    H, W = img2d.shape
    cy, cx = H // 2, W // 2 # center
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X*X + Y*Y)            # radius in pixels
    r = R.astype(np.int32)
    rmax = r.max()

    # bin by integer radius
    sums = np.bincount(r.ravel(), weights=img2d.ravel(), minlength=rmax+1)  # error sum in same radius
    count = np.bincount(r.ravel(), minlength=rmax+1)    # number of pixel in same radius
    count[count == 0] = 1   # avoid 0
    error_per_radius = sums / count                 # mean error per radius-bin

    # normalize radius to [0,1] Nyquist
    f = np.arange(rmax+1) / (rmax if rmax > 0 else 1)
    return f, error_per_radius

@torch.no_grad()
def freq_error_compare(dataset, models, idx, device, savepath='compare_freq.png',
                       vmin=None, vmax=None, eps=1e-8):
    noisy, original = dataset[idx]
    noisy = noisy.unsqueeze(0).to(device)   # (1,C,H,W)
    original = original.unsqueeze(0).to(device)

    error_maps, titles, error_distrubtion = [], [], []
    for model in models:
        model.eval()
        output = model(noisy).clamp(0, 1)
        error  = _freq_error_map(original, output)        # (1,H,W), fftshifted
        error_map = error[0].detach().cpu().numpy()
        error_maps.append(error_map)

        # scalar log-mag MSE
        Lt = _logmag(original, eps)
        Lp = _logmag(output, eps)
        error_log_avg = float(((Lp - Lt)**2).mean())
        titles.append(f'{model.name}\nlog-mag MSE={error_log_avg:.4f}')

        f, error_per_radius = get_error_per_radius(error_map)
        error_distrubtion.append((f, error_per_radius))

    if vmin is None: vmin = min(map(np.min, error_maps))
    if vmax is None: vmax = max(map(np.max, error_maps))

    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 7), constrained_layout=True)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # normalize shape to (2,n)

    last_img = None
    for i, (error_map, title) in enumerate(zip(error_maps, titles)):
        ax = axes[0, i]
        last_img = ax.imshow(error_map, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # shared colorbar for heatmaps
    cbar = fig.colorbar(last_img, ax=axes[0, :], fraction=0.03, pad=0.02)
    cbar.set_label("log-magnitude error", rotation=90)

    # ---- draw radial profiles (log y-scale) ----
    for i, (f, error_per_radius) in enumerate(error_distrubtion):
        ax = axes[1, i]
        # avoid log(0): add tiny epsilon
        ax.plot(f, np.maximum(error_per_radius, eps))
        ax.set_yscale('log')
        ax.set_ylim((10e-6, 10e2))
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("log MSE")

    plt.title(f'Log-magnitude frequency error of im_{idx}.png')
    fig.savefig(savepath, dpi=150)
    return savepath