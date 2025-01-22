import numpy as np
import torch
from torch import nn
from PIL import Image

def load_image(im_path: str, size: tuple[int, int] = (256, 256), device: torch.device | str = torch.device("cpu")) -> torch.Tensor:
    im = Image.open(im_path)
    im = im.convert("L")
    im_numpy = np.array(im)
    im = torch.tensor(im_numpy, device=device).to(torch.float32)
    im = torch.nn.functional.interpolate(im.unsqueeze(0).unsqueeze(0), size, mode='bicubic').squeeze()
    im = im / im.max()
    return im

def sample_image(im: torch.Tensor | nn.Module, coords: torch.Tensor, interp: str = 'bicubic') -> torch.Tensor:
    """
    - im (tensor): image tensor of shape (h, w)
    - im (nn.Module): model to sample image from, takes in coordinates of shape (b, 2)
    - coords (tensor): coordinates to sample image at of shape (n, 2), in range [-1, 1], in format (x, y)
    - interp (str): interpolation method, 'bicubic' | 'bilinear' | 'nearest'
    """
    if isinstance(im, nn.Module): 
        return im(coords)

    assert isinstance(im, torch.Tensor), "im must be tensor or nn.Module"
    assert im.shape[0] == im.shape[1], "Image must be square"
    grid_sample = torch.nn.functional.grid_sample(
        im.unsqueeze(0).unsqueeze(0), coords.unsqueeze(0).unsqueeze(0), 
        align_corners=False, mode=interp, padding_mode='zeros'
        ).squeeze()
    return grid_sample

def create_sinogram(image: torch.Tensor | nn.Module, angles: torch.Tensor, detector_size: int, detector_spacing: float, n_samples: int, inscribe: bool = False) -> torch.Tensor:
    """
    Create sinogram, assume the detector size is normalized to the space where image coordinates are in range [-1, 1]
    - image (tensor): image tensor
    - image (nn.Module): model to sample image from, takes in coordinates of shape (b, 2)
    - angles (tensor): angles to sample sinogram at of shape (n), in radians
    - detector_size (int): size of detector
    - detector_spacing (float): spacing between detectors
    - n_samples (int): number of samples to take for each ray
    - inscribe (bool): if True, the sample will be restricted to the inscribed circle, otherwise the circumscribed
    return sinogram tensor of shape (n, detector_size)
    """
    assert len(angles.shape) == 1, "Angles must be 1D tensor"
    device = angles.device
    n_angles = angles.shape[0]
    # sample coordinates
    def default_sample_coords():
        """
        Create default sample coordinates for sinogram, 
        return tensor of shape (detector_size, n_samples, 2)
        """
        s_coords_x = torch.linspace(-detector_spacing * detector_size / 2, detector_spacing * detector_size / 2, detector_size, device=device)
        if inscribe: s_coords_y = torch.linspace(-1, 1, n_samples, device=device)
        else: s_coords_y = torch.linspace(-np.sqrt(2), np.sqrt(2), n_samples, device=device)
        _s_coords = torch.meshgrid(s_coords_x, s_coords_y, indexing='ij')   # ray for last dimension
        s_coords = torch.stack(_s_coords, dim=-1).reshape(detector_size, n_samples, 2)
        return s_coords
    coords = default_sample_coords()

    def rotate_coords(coords: torch.Tensor, angles: torch.Tensor):
        """
        Rotate coordinates by angle
        - coords (tensor): coordinates to rotate of shape (?, 2)
        - angle (float): angle to rotate by in radians of shape (n)
        return rotated coordinates of shape (n, ?, 2)
        """
        x, y = coords.unbind(-1)
        xx = x.unsqueeze(0).expand(n_angles, -1)
        yy = y.unsqueeze(0).expand(n_angles, -1)
        x_rot = xx * torch.cos(angles).unsqueeze(-1) - yy * torch.sin(angles).unsqueeze(-1)
        y_rot = xx * torch.sin(angles).unsqueeze(-1) + yy * torch.cos(angles).unsqueeze(-1)
        return torch.stack([x_rot, y_rot], dim=-1)
    coords = rotate_coords(coords.view(-1, 2), angles).reshape(n_angles, detector_size, n_samples, 2)

    samples = sample_image(image, coords.view(-1, 2)).reshape(n_angles, detector_size, n_samples)
    return samples.sum(dim=-1)

if __name__ == "__main__":
    def save_im(im: torch.Tensor, path: str):
        im = im.detach().cpu().numpy()
        im = (im - im.min()) / (im.max() - im.min()) * 255
        im = im.astype(np.uint8)
        im = Image.fromarray(im, mode="L")
        im.save(path)

    im = load_image("shepp_logan.png", (256, 256), "cuda")

    # test sampling
    coords = torch.meshgrid(torch.arange(256, device=im.device), torch.arange(256, device=im.device), indexing='xy')
    coords = torch.stack(coords, dim=-1).reshape(-1, 2).to(torch.float32) / 255
    coords = coords * 2 - 1
    new_im = sample_image(im, coords).reshape(256, 256)
    save_im(new_im, "sampled.png")

    # test sinogram
    sino = create_sinogram(im, torch.linspace(0, np.pi, 200).to(im.device), 200, 2/len(im), 256, inscribe=True)
    save_im(sino, "sino.png")