import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BoundedSoftplus(nn.Module):
    """
    BoundedSoftplus activation that prevents extremely large values by clamping.

    Attributes:
        beta (float): Parameter for the softplus function.
        threshold (float): Maximum clamp value.
        eps (float): A small epsilon value added to the final output.
    """

    def __init__(self, beta=1.0, threshold=6.0, eps=1e-6):
        """
        Args:
            beta (float, optional): Beta for F.softplus. Defaults to 1.0.
            threshold (float, optional): Max clamp value. Defaults to 6.0.
            eps (float, optional): Small additive constant. Defaults to 1e-6.
        """
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for the BoundedSoftplus layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after clamped softplus.
        """
        softplus_val = F.softplus(x, beta=self.beta)
        clamped_val = torch.clamp(softplus_val, max=self.threshold)
        return clamped_val + self.eps

class SigmaPredictor(nn.Module):
    """
    Predicts sigma maps (sigma_x, sigma_y, sigma_r) for bilateral filtering
    using a patch-based attention mechanism.

    Attributes:
        patch_size (int): The size of the non-overlapping image patches.
        hidden_dim (int): Dimension for latent representation in attention.
        query, key, value (nn.Linear): Attention transformation for the main features.
        sigma_query, sigma_key, sigma_value (nn.Linear): Attention for sigma features.
        norm (nn.LayerNorm): LayerNorm to normalize the sigma features.
        sigma_proj (nn.Linear): Final linear layer to project to 3 sigma channels.
        activation (BoundedSoftplus): Activation to ensure the sigmas remain positive.
        attention_scale (float): Scale factor for scaled dot-product attention.
    """

    def __init__(self, patch_size=8):
        """
        Args:
            patch_size (int, optional): Patch size for non-overlapping patches. Defaults to 8.
            in_channels (int, optional): Number of channels of the input image. Defaults to 1.
        """
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = 8

        self.query = nn.LazyLinear(self.hidden_dim)
        self.key = nn.LazyLinear(self.hidden_dim)
        self.value = nn.LazyLinear(self.hidden_dim)

        self.sigma_query = nn.LazyLinear(self.hidden_dim)
        self.sigma_key = nn.LazyLinear(self.hidden_dim)
        self.sigma_value = nn.LazyLinear(self.hidden_dim)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.sigma_proj = nn.LazyLinear(3)
        self.activation = BoundedSoftplus(threshold=6)

        self.attention_scale = self.hidden_dim ** -0.5

    def _attention(self, q, k, v):
        """
        Compute scaled dot-product attention.

        Args:
            q (torch.Tensor): Query of shape [B, N, D].
            k (torch.Tensor): Key of shape [B, N, D].
            v (torch.Tensor): Value of shape [B, N, D].

        Returns:
            torch.Tensor: Attention output of shape [B, N, D].
        """
        scores = torch.bmm(q, k.transpose(1, 2)) * self.attention_scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.bmm(attn_weights, v)

    def forward(self, x):
        """
        Predict sigmas from the input image.

        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted sigma map of shape (B, H, W, 3).
        """
        b, c, h, w = x.shape
        ps = self.patch_size
        assert h % ps == 0 and w % ps == 0, "Image dimensions must be divisible by patch_size"

        # Extract non-overlapping patches
        patches = x.view(b, c, h // ps, ps, w // ps, ps)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(b, (h // ps) * (w // ps), -1)

        # Feature attention
        q_main = self.query(patches)
        k_main = self.key(patches)
        v_main = self.value(patches)
        feats = self._attention(q_main, k_main, v_main)

        # Sigma attention
        q_sigma = self.sigma_query(feats)
        k_sigma = self.sigma_key(feats)
        v_sigma = self.sigma_value(feats)
        sigmas_patch = self._attention(q_sigma, k_sigma, v_sigma)
        sigmas_patch_norm = self.norm(sigmas_patch)
        sigmas_patch_proj = self.sigma_proj(sigmas_patch_norm)
        sigmas = self.activation(sigmas_patch_proj)  # [B, N, 3]

        # Reshape -> [B, 3, H//ps, W//ps]
        sigmas = sigmas.view(b, h // ps, w // ps, 3).permute(0, 3, 1, 2)

        # Upsample to original resolution via nearest neighbor
        sigmas_resized = F.interpolate(sigmas, size=(h, w), mode='nearest').permute(0, 2, 3, 1)
        return sigmas_resized

class AGBF(nn.Module):
    """
    Attention-Guided Bilateral Filtering (AGBF) module that uses predicted sigmas
    from SigmaPredictor. Combines both spatial and range kernels in a single pass.

    This class also caches grid coordinates for performance.

    Attributes:
        _cached_grids (dict): Class-level cache for grid coordinates.
        sigma_predictor (SigmaPredictor): Predicts the sigma maps.
    """
    _cached_grids = {}

    def __init__(self, patch_size=16):
        """
        Args:
            patch_size (int, optional): Patch size for the SigmaPredictor. Defaults to 16.
            in_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__()
        self.sigma_predictor = SigmaPredictor(patch_size=patch_size)

    def compute_spatial_kernel(self, sx, sy, k, device):
        """
        Compute the 2D spatial Gaussian kernel.

        Args:
            sx (torch.Tensor): Sigma_x values of shape [B, H, W].
            sy (torch.Tensor): Sigma_y values of shape [B, H, W].
            k (int): Kernel size.
            device (torch.device): Device to use.

        Returns:
            torch.Tensor: Spatial kernel of shape [B, 1, H, W, k*k].
        """
        if k not in AGBF._cached_grids:
            half_k = k // 2
            y_coords, x_coords = torch.meshgrid(
                torch.arange(-half_k, half_k + 1, device=device),
                torch.arange(-half_k, half_k + 1, device=device),
                indexing='ij'
            )
            x_coords = x_coords.float().view(1, 1, 1, k, k)
            y_coords = y_coords.float().view(1, 1, 1, k, k)
            AGBF._cached_grids[k] = (x_coords, y_coords)
        else:
            x_coords, y_coords = AGBF._cached_grids[k]
            if x_coords.device != device:
                x_coords = x_coords.to(device)
                y_coords = y_coords.to(device)

        b, h, w = sx.shape
        sx_expanded = sx.view(b, h, w, 1, 1)
        sy_expanded = sy.view(b, h, w, 1, 1)

        spatial_kernel = torch.exp(
            -((x_coords**2) / (2 * sx_expanded**2) + (y_coords**2) / (2 * sy_expanded**2))
        )
        return spatial_kernel

    def compute_range_kernel(self, center_values, neighbor_values, sigma_r):
        """
        Compute the range kernel for bilateral filtering.

        Args:
            center_values (torch.Tensor): Center pixel values, shape [B, C, H, W, 1].
            neighbor_values (torch.Tensor): Neighboring pixel values, shape [B, C, H, W, k*k].
            sigma_r (torch.Tensor): Range sigma values, shape [B, H, W].

        Returns:
            torch.Tensor: Range kernel, shape [B, C, H, W, k*k].
        """
        diff = center_values - neighbor_values
        sq_diff = diff**2
        if center_values.shape[1] > 1:
            sq_diff = sq_diff.sum(dim=1, keepdim=True)  # Sum across channels if multi-channel

        sigma_r_expanded = sigma_r.unsqueeze(1).unsqueeze(-1)
        range_kernel = torch.exp(-sq_diff / (2 * sigma_r_expanded**2))
        return range_kernel

    def forward(self, x, return_sigmas=False):
        """
        Forward pass for the AGBF module. Predicts sigmas, computes bilateral filtering.

        Args:
            x (torch.Tensor): Input of shape [B, C, H, W].
            return_sigmas (bool, optional): If True, also return the predicted sigmas.

        Returns:
            torch.Tensor: Filtered output of shape [B, C, H, W].
            (Optional) torch.Tensor: Sigma map of shape [B, H, W, 3] if return_sigmas=True.
        """
        x = x.float()
        b, c, h, w = x.shape

        # Predict sigma maps
        s = self.sigma_predictor(x)
        sx = s[..., 0]
        sy = s[..., 1]
        sr = s[..., 2]

        # Determine kernel size dynamically
        m = torch.max(sx.max(), sy.max()).item()
        k = int(2 * math.ceil(m) + 1)
        if k % 2 == 0:
            k += 1
        p = k // 2

        # Pad the input
        xp = F.pad(x, (p, p, p, p), mode='reflect')
        patches = F.unfold(xp, kernel_size=k, stride=1).view(b, c, k*k, h, w).permute(0, 1, 3, 4, 2)

        # Compute spatial kernel
        spatial_kernel = self.compute_spatial_kernel(sx, sy, k, x.device)
        spatial_kernel = spatial_kernel.view(b, 1, h, w, k * k)

        # Compute range kernel
        center_values = x.view(b, c, h, w, 1)
        range_kernel = self.compute_range_kernel(center_values, patches, sr)

        # Combine kernels
        combined_kernel = spatial_kernel * range_kernel
        norm_factor = combined_kernel.sum(dim=-1, keepdim=True)
        normalized_kernel = combined_kernel / (norm_factor + 1e-8)

        # Apply filter
        patches.mul_(normalized_kernel)
        output = patches.sum(dim=-1)

        if return_sigmas:
            return output, s
        return output