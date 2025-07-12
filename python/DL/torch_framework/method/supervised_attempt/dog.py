import torch
import torch.nn as nn
import torch.nn.functional as F

class F2N_edge:
    """
    A self-supervised loss function for image denoising that combines:
      - Multi-scale consistency
      - Edge preservation using a Difference of Gaussians (DoG) filter

    Attributes:
        kernel_size (int): Size of the DoG kernel.
        sigma_narrow (float): Narrow Gaussian sigma.
        sigma_wide (float): Wide Gaussian sigma.
        device (torch.device): Target device.
        edge_loss_weight (float): Weight for the edge loss term.
        base_dog_kernel (torch.Tensor): The DoG kernel used for edge preservation.
    """

    def __init__(self, device, lambda_=350):
        """
        Args:
            device (torch.device): Device for kernel computations.
            lambda_ (float, optional): Weight for the edge loss term. Defaults to 350.
        """
        self.kernel_size = 7
        self.sigma_narrow = 9
        self.sigma_wide = 10
        self.device = device
        self.edge_loss_weight = lambda_
        self._initialize_dog_kernel()

    def _initialize_dog_kernel(self):
        """
        Create a Difference of Gaussians (DoG) kernel used for edge detection.
        """
        grid_coords = torch.arange(self.kernel_size, dtype=torch.float32,
                                   device=self.device) - self.kernel_size // 2
        grid_y, grid_x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')

        gaussian_narrow = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma_narrow**2))
        gaussian_narrow = gaussian_narrow / gaussian_narrow.sum()

        gaussian_wide = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma_wide**2))
        gaussian_wide = gaussian_wide / gaussian_wide.sum()

        dog_kernel = (gaussian_narrow - gaussian_wide).unsqueeze(0).unsqueeze(0)
        self.base_dog_kernel = dog_kernel

    def __call__(self, a, b, alpha=None):
        """
        Calculate the self-supervised loss on the given noisy image and model outputs.

        Args:
            noisy_input (torch.Tensor): A noisy image of shape (B, C, H, W).
            model (nn.Module): The denoising model.
            alpha (float, optional): If given, overrides the edge_loss_weight.

        Returns:
            torch.Tensor: The total loss value.
        """
        

        dog_kernel = self.base_dog_kernel.repeat(a.shape[1], 1, 1, 1)


        # Edge preservation using DoG
        edges_noisy = F.conv2d(a, dog_kernel, padding=self.kernel_size // 2,
                               groups=a.shape[1])
        edges_denoised = F.conv2d(b, dog_kernel, padding=self.kernel_size // 2,
                                  groups=a.shape[1])
        loss_edge = self.edge_loss_weight * F.l1_loss(
            torch.abs(edges_noisy), torch.abs(edges_denoised)
        )

        return loss_edge