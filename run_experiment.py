import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from wavefront_coding import (
    make_grid, compute_psf, compute_mtf,
    simulate_capture, restore_wiener, restore_richardson_lucy,
    evaluate,
)

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def part1_psf_and_mtf_comparison():
    """Compare PSF/MTF stability across defocus, with vs without mask."""
    X, Y, aperture = make_grid(n=256, aperture_radius=0.5)

    psi_values = np.linspace(0, 20 * np.pi, 5)   # defocus levels
    alpha_no_mask = 0.0
    alpha_with_mask = 30 * np.pi                 # mask strength (tune this!)

    fig, axes = plt.subplots(2, len(psi_values), figsize=(3 * len(psi_values), 6))

    for i, psi in enumerate(psi_values):
        psf_no_mask = compute_psf(X, Y, aperture, psi, alpha_no_mask)
        psf_with_mask = compute_psf(X, Y, aperture, psi, alpha_with_mask)

        axes[0, i].imshow(psf_no_mask, cmap="inferno")
        axes[0, i].set_title(f"no mask\npsi={psi:.1f}")
        axes[0, i].axis("off")

        axes[1, i].imshow(psf_with_mask, cmap="inferno")
        axes[1, i].set_title(f"with mask\npsi={psi:.1f}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("no mask")
    axes[1, 0].set_ylabel("with mask")
    fig.suptitle("PSF vs defocus: conventional system (top) vs "
                 "wavefront-coded system (bottom)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_psf_comparison.png"), dpi=150)
    plt.close(fig)

    # MTF comparison at the largest defocus (worst case)
    psi_worst = psi_values[-1]
    psf_no_mask = compute_psf(X, Y, aperture, psi_worst, alpha_no_mask)
    psf_with_mask = compute_psf(X, Y, aperture, psi_worst, alpha_with_mask)
    mtf_no_mask = compute_mtf(psf_no_mask)
    mtf_with_mask = compute_mtf(psf_with_mask)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mtf_no_mask, cmap="viridis")
    axes[0].set_title("MTF, no mask (worst-case defocus)")
    axes[0].axis("off")
    axes[1].imshow(mtf_with_mask, cmap="viridis")
    axes[1].set_title("MTF, with mask (worst-case defocus)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_mtf_comparison.png"), dpi=150)
    plt.close(fig)

    print("[part1] Saved PSF and MTF comparison figures.")
    return X, Y, aperture, alpha_with_mask


def part2_image_simulation_and_restoration(X, Y, aperture, alpha_with_mask):
    """Simulate a coded capture and restore it with two deconvolution
    methods, then evaluate quality."""

    # Ground-truth test image
    image = data.camera() / 255.0
    image = image.astype(np.float64)

    # Pick a mid-range defocus to simulate
    psi_test = 10 * np.pi
    psf_test = compute_psf(X, Y, aperture, psi_test, alpha_with_mask)

    # Also compute the equivalent conventional (no-mask) blur for comparison
    psf_no_mask_test = compute_psf(X, Y, aperture, psi_test, 0.0)

    blurred_coded = simulate_capture(image, psf_test, noise_sigma=0.01)
    blurred_conventional = simulate_capture(image, psf_no_mask_test, noise_sigma=0.01)

    restored_wiener = restore_wiener(blurred_coded, psf_test, balance=0.1)
    restored_rl = restore_richardson_lucy(blurred_coded, psf_test, num_iter=30)

    psnr_w, ssim_w = evaluate(image, restored_wiener)
    psnr_rl, ssim_rl = evaluate(image, restored_rl)

    # Save a comparison figure: truth / conventional blur / coded raw / restored
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles_images = [
        ("Ground truth", image),
        ("Conventional system\n(blurred, no mask)", blurred_conventional),
        ("Coded raw capture\n(with mask, before deconv)", blurred_coded),
        (f"Restored: Wiener\nPSNR={psnr_w:.1f} SSIM={ssim_w:.3f}", restored_wiener),
        (f"Restored: Richardson-Lucy\nPSNR={psnr_rl:.1f} SSIM={ssim_rl:.3f}", restored_rl),
    ]
    for ax, (title, im) in zip(axes, titles_images):
        ax.imshow(np.clip(im, 0, 1), cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_restoration_comparison.png"), dpi=150)
    plt.close(fig)

    # Save numeric results to a text file
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write("Restoration quality metrics\n")
        f.write("============================\n")
        f.write(f"Wiener filter          : PSNR = {psnr_w:.2f} dB, SSIM = {ssim_w:.4f}\n")
        f.write(f"Richardson-Lucy (30 it): PSNR = {psnr_rl:.2f} dB, SSIM = {ssim_rl:.4f}\n")

    print("[part2] Saved restoration comparison figure and metrics.txt")
    print(f"  Wiener          -> PSNR={psnr_w:.2f} dB, SSIM={ssim_w:.4f}")
    print(f"  Richardson-Lucy -> PSNR={psnr_rl:.2f} dB, SSIM={ssim_rl:.4f}")


def main():
    X, Y, aperture, alpha_with_mask = part1_psf_and_mtf_comparison()
    part2_image_simulation_and_restoration(X, Y, aperture, alpha_with_mask)
    print(f"\nAll results saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
