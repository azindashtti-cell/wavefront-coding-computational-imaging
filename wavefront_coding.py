import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import wiener, richardson_lucy


# ---------------------------------------------------------------------
# 1. Pupil / grid setup
# ---------------------------------------------------------------------

def make_grid(n=256, aperture_radius=0.5):
    """Create a normalized coordinate grid and circular aperture mask.

    Parameters
    ----------
    n : int
        Grid size (n x n).
    aperture_radius : float
        Radius of the circular aperture in normalized coordinates
        (grid spans [-1, 1]).

    Returns
    -------
    X, Y : 2D arrays of normalized coordinates
    aperture : boolean 2D array, True inside the aperture
    """
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    aperture = (X**2 + Y**2) <= aperture_radius**2
    return X, Y, aperture


# ---------------------------------------------------------------------
# 2. PSF computation
# ---------------------------------------------------------------------

def compute_psf(X, Y, aperture, psi, alpha):
    """Compute the normalized PSF for given defocus (psi) and cubic
    phase mask strength (alpha).

    Parameters
    ----------
    X, Y : coordinate grids from make_grid()
    aperture : boolean aperture mask from make_grid()
    psi : float
        Defocus strength. psi = 0 means in-focus.
    alpha : float
        Cubic phase mask strength. alpha = 0 means no mask
        (conventional optical system).

    Returns
    -------
    psf : 2D array, energy-normalized (sums to 1)
    """
    phase_defocus = psi * (X**2 + Y**2)
    phase_cubic = alpha * (X**3 + Y**3)
    phase_total = phase_defocus + phase_cubic

    pupil = aperture * np.exp(1j * phase_total)
    psf = np.abs(fftshift(fft2(pupil))) ** 2
    psf = psf / psf.sum()
    return psf


def compute_mtf(psf):
    """Modulation Transfer Function magnitude from a PSF."""
    mtf = np.abs(fftshift(fft2(psf)))
    return mtf / mtf.max()


# ---------------------------------------------------------------------
# 3. Forward model: simulate a raw (blurred, noisy) capture
# ---------------------------------------------------------------------

def simulate_capture(image, psf, noise_sigma=0.01, seed=0):
    """Convolve a ground-truth image with a PSF and add Gaussian noise.

    Parameters
    ----------
    image : 2D float array in [0, 1]
    psf : 2D array, same shape as image (or smaller; will be padded)
    noise_sigma : standard deviation of additive Gaussian noise
    seed : RNG seed for reproducibility

    Returns
    -------
    blurred : 2D float array, the simulated raw sensor capture
    """
    rng = np.random.default_rng(seed)

    # Pad / crop psf to match image shape, then convolve via FFT
    psf_padded = _match_shape(psf, image.shape)
    psf_padded = psf_padded / psf_padded.sum()

    image_f = fft2(image)
    psf_f = fft2(fftshift(psf_padded))
    blurred = np.real(ifft2(image_f * psf_f))

    blurred += rng.normal(0, noise_sigma, image.shape)
    blurred = np.clip(blurred, 0, 1)
    return blurred


def _match_shape(psf, target_shape):
    """Center-crop or zero-pad psf to match target_shape."""
    out = np.zeros(target_shape, dtype=psf.dtype)
    ph, pw = psf.shape
    th, tw = target_shape

    # crop if psf is larger
    if ph > th:
        start = (ph - th) // 2
        psf = psf[start:start + th, :]
        ph = th
    if pw > tw:
        start = (pw - tw) // 2
        psf = psf[:, start:start + tw]
        pw = tw

    # place (possibly smaller) psf centered in the output
    y0 = (th - ph) // 2
    x0 = (tw - pw) // 2
    out[y0:y0 + ph, x0:x0 + pw] = psf
    return out


# ---------------------------------------------------------------------
# 4. Restoration (deconvolution)
# ---------------------------------------------------------------------

def restore_wiener(blurred, psf, balance=0.1):
    """Wiener deconvolution using the known PSF."""
    psf_norm = psf / psf.sum()
    return wiener(blurred, psf_norm, balance=balance)


def restore_richardson_lucy(blurred, psf, num_iter=30):
    """Richardson-Lucy iterative deconvolution using the known PSF."""
    psf_norm = psf / psf.sum()
    return richardson_lucy(blurred, psf_norm, num_iter=num_iter)


# ---------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------

def evaluate(reference, estimate):
    """Return (PSNR, SSIM) between reference and estimate, both in [0,1]."""
    estimate = np.clip(estimate, 0, 1)
    p = psnr(reference, estimate, data_range=1.0)
    s = ssim(reference, estimate, data_range=1.0)
    return p, s
