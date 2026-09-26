import torch
import torch.nn as nn
from typing import Tuple, Union

class TorchRunningMeanStd(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Union[int, Tuple[int, ...]] = (),
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        if isinstance(shape, int):
            shape = (shape,)

        self.epsilon = epsilon
        self.stats_dtype = dtype

        self.register_buffer("mean", torch.zeros(shape, device=device, dtype=dtype))
        self.register_buffer("var", torch.ones(shape, device=device, dtype=dtype))
        self.register_buffer("count", torch.tensor(epsilon, device=device, dtype=dtype))

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        """
        arr: [B, obs_dim] or [B, ...]
        """
        arr = arr.detach().to(device=self.mean.device, dtype=self.stats_dtype)

        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = torch.tensor(arr.shape[0], device=self.mean.device, dtype=self.stats_dtype)

        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count

        new_var = m_2 / total_count
        new_count = total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(new_count)


class TorchNormalizer(TorchRunningMeanStd):
    def __init__(
        self,
        input_dim,
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
        device: str = "cuda:0",
    ):
        super().__init__(
            epsilon=epsilon,
            shape=input_dim,
            device=device,
            dtype=torch.float64,
        )
        self.norm_epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize_torch(self, input: torch.Tensor, device) -> torch.Tensor:

        mean = self.mean.to(dtype=input.dtype)
        std = torch.sqrt((self.var + self.norm_epsilon).to(dtype=input.dtype))

        return torch.clamp(
            (input - mean) / std,
            -self.clip_obs,
            self.clip_obs,
        )

    def normalize(self, input):
        """
        如果你还有 numpy 代码需要兼容，可以保留这个接口。
        训练主流程不建议用它。
        """
        input_t = torch.as_tensor(input, device=self.mean.device, dtype=torch.float32)
        out = self.normalize_torch(input_t)
        return out.cpu().numpy()