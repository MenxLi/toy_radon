import pathlib
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from toy_radon import create_sinogram, sample_image, load_image

class DifferentiableImage(nn.Module):
    def __init__(self, size: tuple[int, int]):
        super().__init__()
        self.image = nn.Parameter(torch.rand(size))
    def forward(self, x):
        return sample_image(self.image, x)
    def tv(self):
        return (self.image[:, :-1] - self.image[:, 1:]).pow(2).mean() + (self.image[:-1] - self.image[1:]).pow(2).mean()

def vis(model: nn.Module, size: tuple[int, int], device: torch.device = torch.device("cuda")):
    assert size[0] == size[1]
    s_coords = torch.linspace(-1, 1, size[0], device=device)
    coords = torch.meshgrid(s_coords, s_coords, indexing='xy')
    coords = torch.stack(coords, dim=-1).reshape(-1, 2)
    with torch.no_grad():
        return sample_image(model, coords).reshape(size)

def train(image: torch.Tensor, angles: torch.Tensor, output_dir: pathlib.Path):

    model = DifferentiableImage((256, 256)).to(image.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
    criterion = nn.MSELoss()
    sinogram = create_sinogram(image, angles, 256, 2/len(image), 256)

    n_iter = 20000
    for i in range(n_iter):
        optimizer.zero_grad()
        ang_idx = torch.randint(0, len(angles), (1,))
        model_sino = create_sinogram(model, angles[ang_idx], 256, 2/len(image), 256)
        aim = sinogram[ang_idx]
        loss = criterion(model_sino, aim) + model.tv() * 10
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Step {i}, Loss: {loss.item()}")
            im = vis(model, (256, 256))
            plt.imsave(output_dir / f"iter_{i:05d}.png", im.detach().cpu().numpy(), cmap="gray")
        
        if i == n_iter // 2:
            optimizer.param_groups[0]["lr"] = 3e-4
    
    plt.imsave(output_dir / "final.png", vis(model, (256, 256)).detach().cpu().numpy(), cmap="gray")
    return model

if __name__ == "__main__":
    output_dir = pathlib.Path("output")
    output_dir.mkdir(exist_ok=True)
    image = load_image("shepp_logan.png", (256, 256), "cuda")
    train(image, torch.linspace(0, torch.pi, 180).to(image.device), output_dir)
