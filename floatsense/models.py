# pylint: disable=too-many-lines
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-callable
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-branches
# pylint: disable=unused-argument
"""Sequence models for acceleration-to-moment reconstruction.

Two families:
  - `SpectralGainModel`: a learned, operating-state-conditioned transfer
    function. A per-bin complex gain is applied to the rFFT of the
    acceleration; a hypernetwork adds a condition-dependent correction to
    the base gain. This is the learned generalization of the physics
    baseline's band gains (interpretable: the gain magnitude can be plotted
    against the calibrated C_theta curve).
  - `TCNModel`: a dilated temporal convolutional network over the
    acceleration and time-varying operating-state channels, length-agnostic
    (train on crops, evaluate on full series).
"""

from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn


class SpectralGainModel(nn.Module):
    """Condition-dependent complex gain in the frequency domain.

    forward(inputs, condition) uses only the first input channel (the
    acceleration); the operating state enters through the condition vector.
    The sequence length must be fixed (`num_samples`).
    """

    def __init__(self,
                 num_samples: int,
                 condition_dim: int = 3,
                 hidden_dim: int = 64):
        """Initializes the model.

        Args:
            num_samples (int): Fixed input sequence length.
            condition_dim (int): Number of condition scalars.
            hidden_dim (int): Hidden width of the conditioning network.
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_bins = num_samples // 2 + 1
        self.base_gain = nn.Parameter(torch.zeros(self.num_bins, 2))
        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_bins * 2),
        )
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        accel = inputs[:, 0, :]
        spectrum = torch.fft.rfft(accel, dim=-1)
        gain = self.base_gain[None, :, :] + self.condition_net(condition).view(
            -1, self.num_bins, 2)
        gain = torch.complex(gain[..., 0], gain[..., 1])
        moment = torch.fft.irfft(spectrum * gain, n=accel.shape[-1], dim=-1)
        return moment[:, None, :]

    def gain_magnitude(self, condition: torch.Tensor) -> torch.Tensor:
        """Returns |gain| per frequency bin for a batch of conditions."""
        gain = self.base_gain[None, :, :] + self.condition_net(condition).view(
            -1, self.num_bins, 2)
        return torch.sqrt(gain[..., 0]**2 + gain[..., 1]**2)


class HybridGainModel(nn.Module):
    """Physics-anchored spectral gain: learned correction to physics gains.

    The reconstruction applies G_phys(f) * (1 + c(f, condition)) to the
    rFFT of the acceleration, where G_phys is the per-simulation band gain
    of the calibrated physics baseline (supplied by the dataset as
    'physics_gain') and c is a complex learned correction initialized at
    zero, so training starts exactly at the physics reconstruction. The
    condition-dependent part of c is bounded with a tanh (scaled by
    `condition_bound`): an unbounded hypernetwork extrapolates as badly as
    the plain spectral model and destroys the physics anchor out of
    distribution, while the bounded correction keeps the gain within
    (1 +- bound) of the physics by construction.
    """

    needs_physics_gain = True

    def __init__(self,
                 num_samples: int,
                 condition_dim: int = 3,
                 hidden_dim: int = 64,
                 condition_bound: float = 0.5):
        """Initializes the model.

        Args:
            num_samples (int): Fixed input sequence length.
            condition_dim (int): Number of condition scalars.
            hidden_dim (int): Hidden width of the conditioning network.
            condition_bound (float): Maximum magnitude of each component of
              the condition-dependent correction (0 disables the bound).
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_bins = num_samples // 2 + 1
        self.condition_bound = condition_bound
        self.base_correction = nn.Parameter(torch.zeros(self.num_bins, 2))
        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_bins * 2),
        )
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

    def _correction(self, condition: torch.Tensor) -> torch.Tensor:
        conditioned = self.condition_net(condition).view(-1, self.num_bins, 2)
        if self.condition_bound:
            conditioned = self.condition_bound * torch.tanh(conditioned)
        correction = self.base_correction[None, :, :] + conditioned
        return torch.complex(correction[..., 0], correction[..., 1])

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor,
                physics_gain: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        accel = inputs[:, 0, :]
        spectrum = torch.fft.rfft(accel, dim=-1)
        gain = physics_gain.to(
            spectrum.dtype) * (1.0 + self._correction(condition))
        moment = torch.fft.irfft(spectrum * gain, n=accel.shape[-1], dim=-1)
        return moment[:, None, :]

    def gain_magnitude(self, condition: torch.Tensor,
                       physics_gain: torch.Tensor) -> torch.Tensor:
        """Returns |gain| per frequency bin for a batch of conditions."""
        gain = physics_gain.to(
            torch.complex64) * (1.0 + self._correction(condition))
        return torch.abs(gain)


class _TCNBlock(nn.Module):
    """Residual block with two dilated causal-padded convolutions."""

    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels,
                               channels,
                               kernel_size,
                               padding=padding,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(channels,
                               channels,
                               kernel_size,
                               padding=padding,
                               dilation=dilation)
        self.activation = nn.GELU()
        # Identity at 0 (no parameters, no random draw): the published block.
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the residual block to (batch, channels, length) inputs."""
        hidden = self.dropout(self.activation(self.conv1(inputs)))
        hidden = self.conv2(hidden)
        return self.activation(inputs + hidden)


class TCNModel(nn.Module):
    """Dilated temporal convolutional network, length-agnostic."""

    def __init__(self,
                 input_channels: int = 4,
                 hidden_channels: int = 64,
                 kernel_size: int = 5,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
                 dropout: float = 0.0):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels (acceleration +
              time-varying condition channels).
            hidden_channels (int): Width of the residual blocks.
            kernel_size (int): Convolution kernel size.
            dilations (Sequence[int]): Dilation of each residual block.
            dropout (float): Dropout after the first convolution of each
              block (0 = none, as published).
        """
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden_channels, kernel_size, dilation, dropout)
            for dilation in dilations
        ])
        self.output_proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The operating state enters as time-varying channels.
        hidden = self.input_proj(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_proj(hidden)


class LSTMModel(nn.Module):
    """Bidirectional LSTM over acceleration and state channels."""

    def __init__(self,
                 input_channels: int = 4,
                 hidden_size: int = 96,
                 num_layers: int = 2,
                 dropout: float = 0.0):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            hidden_size (int): LSTM hidden size (per direction).
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout between stacked layers (0 = none, as
              published; ignored with one layer, as torch has none there).
        """
        super().__init__()
        self.lstm = nn.LSTM(input_channels,
                            hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_proj = nn.Linear(2 * hidden_size, 1)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The operating state enters as time-varying channels.
        hidden, _ = self.lstm(inputs.transpose(1, 2))
        return self.output_proj(hidden).transpose(1, 2)


class PatchTransformerModel(nn.Module):
    """Transformer encoder over strided temporal patches.

    The series is embedded into non-overlapping patches (strided conv), a
    learned positional embedding is added, and a linear head maps each
    encoded token back to its patch of moment samples.
    """

    def __init__(self,
                 input_channels: int = 4,
                 patch_size: int = 16,
                 embed_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 max_tokens: int = 1024,
                 dropout: float = 0.1):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            patch_size (int): Samples per non-overlapping patch.
            embed_dim (int): Token embedding width.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Attention heads per layer.
            max_tokens (int): Maximum number of patches (positional table).
            dropout (float): Dropout of the encoder layers (0.1, the torch
              default, as published).
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Conv1d(input_channels,
                               embed_dim,
                               patch_size,
                               stride=patch_size)
        self.positional = nn.Parameter(0.02 *
                                       torch.randn(1, max_tokens, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim,
                                           num_heads,
                                           dim_feedforward=4 * embed_dim,
                                           dropout=dropout,
                                           batch_first=True,
                                           activation="gelu",
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(embed_dim, patch_size)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The operating state enters as time-varying channels.
        length = inputs.shape[-1]
        padding = (-length) % self.patch_size
        if padding:
            inputs = nn.functional.pad(inputs, (0, padding))
        tokens = self.embed(inputs).transpose(1, 2)
        tokens = tokens + self.positional[:, :tokens.shape[1], :]
        encoded = self.encoder(tokens)
        moment = self.head(encoded).reshape(encoded.shape[0], 1, -1)
        return moment[..., :length]


class _S4DLayer(nn.Module):
    """Diagonal state-space layer (S4D) with ZOH discretization.

    The SSM kernel is materialized in the time domain from the diagonal
    state matrix and applied by FFT convolution, so the layer is
    length-agnostic (train on crops, evaluate on full windows).
    """

    def __init__(self, channels: int, state_dim: int = 32):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        log_dt = torch.rand(channels) * (torch.log(
            torch.tensor(0.1)) - torch.log(torch.tensor(0.001))) + torch.log(
                torch.tensor(0.001))
        self.log_dt = nn.Parameter(log_dt)
        self.log_a_real = nn.Parameter(
            torch.log(0.5 * torch.ones(channels, state_dim)))
        self.a_imag = nn.Parameter(
            torch.pi * torch.arange(state_dim).float().repeat(channels, 1))
        self.c = nn.Parameter(torch.randn(channels, state_dim, 2) * 0.5**0.5)
        self.d = nn.Parameter(torch.ones(channels))

    def _kernel(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)[:, None]
        a = -torch.exp(self.log_a_real) + 1j * self.a_imag
        c = torch.complex(self.c[..., 0], self.c[..., 1])
        dt_a = a * dt
        c_scaled = c * (torch.exp(dt_a) - 1.0) / a
        positions = torch.arange(length, device=dt.device)
        vandermonde = torch.exp(dt_a[..., None] * positions)
        return 2 * torch.einsum("cn,cnl->cl", c_scaled, vandermonde).real

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the SSM to (batch, channels, length) inputs."""
        length = inputs.shape[-1]
        kernel = self._kernel(length)
        fft_len = 2 * length
        spectrum = torch.fft.rfft(inputs, n=fft_len) * torch.fft.rfft(kernel,
                                                                      n=fft_len)
        output = torch.fft.irfft(spectrum, n=fft_len)[..., :length]
        return output + self.d[:, None] * inputs


class _S4DBlock(nn.Module):
    """Residual S4D block: norm -> SSM -> GELU -> pointwise mix."""

    def __init__(self, channels: int, state_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ssm = _S4DLayer(channels, state_dim)
        self.activation = nn.GELU()
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the block to (batch, channels, length) inputs."""
        hidden = self.norm(inputs.transpose(1, 2)).transpose(1, 2)
        hidden = self.mix(self.activation(self.ssm(hidden)))
        return inputs + hidden


class S4Model(nn.Module):
    """Stack of diagonal state-space (S4D) blocks, length-agnostic.

    State-space models carry an inductive bias close to linear dynamical
    systems (each state is a damped oscillator), which is the natural
    hypothesis class for structural response.
    """

    def __init__(self,
                 input_channels: int = 4,
                 hidden_channels: int = 64,
                 state_dim: int = 32,
                 num_layers: int = 4):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Width of the residual blocks.
            state_dim (int): States per channel in each SSM layer.
            num_layers (int): Number of S4D blocks.
        """
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [_S4DBlock(hidden_channels, state_dim) for _ in range(num_layers)])
        self.output_proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The state enters through the temporal channels.
        hidden = self.input_proj(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_proj(hidden)


class _SpectralConv(nn.Module):
    """Fourier layer: truncated complex multiplication in frequency."""

    def __init__(self, channels: int, num_modes: int):
        super().__init__()
        self.num_modes = num_modes
        scale = 1.0 / channels
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, num_modes, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the Fourier layer to (batch, channels, length)."""
        length = inputs.shape[-1]
        spectrum = torch.fft.rfft(inputs, dim=-1)
        modes = min(self.num_modes, spectrum.shape[-1])
        weight = torch.complex(self.weight[..., 0], self.weight[..., 1])
        out = torch.zeros_like(spectrum)
        out[..., :modes] = torch.einsum("bcm,com->bom", spectrum[..., :modes],
                                        weight[..., :modes])
        return torch.fft.irfft(out, n=length, dim=-1)


class FNOModel(nn.Module):
    """Fourier neural operator over the input channels.

    Learns a resolution-independent operator between the input signals and
    the moment series by mixing a truncated set of Fourier modes, the
    natural operator-learning baseline for this task.
    """

    def __init__(self,
                 input_channels: int = 4,
                 hidden_channels: int = 48,
                 num_modes: int = 128,
                 num_layers: int = 4):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Width of the lifted representation.
            num_modes (int): Fourier modes kept per layer.
            num_layers (int): Number of Fourier layers.
        """
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, 1)
        self.spectral = nn.ModuleList([
            _SpectralConv(hidden_channels, num_modes) for _ in range(num_layers)
        ])
        self.pointwise = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, 1)
            for _ in range(num_layers)
        ])
        self.activation = nn.GELU()
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1), nn.GELU(),
            nn.Conv1d(hidden_channels, 1, 1))

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The state enters through the temporal channels.
        hidden = self.input_proj(inputs)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            hidden = self.activation(spectral(hidden) + pointwise(hidden))
        return self.output_proj(hidden)


class ChronosEncoderModel(nn.Module):
    """Frozen time-series foundation model as encoder, trained linear head.

    Chronos-T5 is a univariate forecaster, so it is used here the standard
    way a foundation model enters a regression benchmark: the encoder is
    frozen and only a head that maps its token embeddings back to the
    moment series is trained. The input is processed in fixed context
    chunks; Chronos rescales each chunk internally, so the head also
    receives the chunk scale to recover absolute amplitude.
    """

    def __init__(self,
                 input_channels: int = 4,
                 context_length: int = 512,
                 model_name: str = "amazon/chronos-t5-small",
                 finetune: bool = False):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels (only the first,
              the acceleration, is embedded).
            context_length (int): Chronos context window in samples.
            model_name (str): Hugging Face id of the Chronos checkpoint.
            finetune (bool): Train the encoder as well as the head.
        """
        super().__init__()
        from chronos import ChronosPipeline  # pylint: disable=import-outside-toplevel
        self.context_length = context_length
        self.pipeline = ChronosPipeline.from_pretrained(model_name,
                                                        dtype=torch.float32)
        self.encoder = self.pipeline.model.model.encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(finetune)
        width = self.encoder.config.d_model
        self.head = nn.Sequential(
            nn.Conv1d(width + 1, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 1, 1),
        )

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition
        accel = inputs[:, 0, :]
        batch, length = accel.shape
        chunk = self.context_length
        pad = (-length) % chunk
        if pad:
            accel = torch.nn.functional.pad(accel, (0, pad))
        chunks = accel.view(batch, -1, chunk).reshape(-1, chunk)
        scale = chunks.abs().mean(dim=-1, keepdim=True) + 1e-6
        with torch.no_grad():
            tokens, _ = self.pipeline.embed(chunks.cpu())
        tokens = tokens[:, :chunk, :].to(accel.device).transpose(1, 2)
        scale_row = scale.to(accel.device)[:, :, None].expand(-1, 1, chunk)
        moment = self.head(torch.cat([tokens, scale_row], dim=1))
        moment = moment.view(batch, 1, -1)[:, :, :length]
        return moment


class MomentModel(nn.Module):
    """MOMENT foundation model with a trainable reconstruction head.

    MOMENT is pretrained for masked time-series reconstruction, so it maps
    a window to a window of the same length. Here the patch encoder is
    frozen and only the head is trained, which turns the pretrained
    representation into the moment series: the standard way a time-series
    foundation model enters a regression benchmark.
    """

    def __init__(self,
                 input_channels: int = 4,
                 context_length: int = 512,
                 model_name: str = "AutonLab/MOMENT-1-small",
                 finetune: bool = False):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels (only the first,
              the acceleration, is encoded).
            context_length (int): MOMENT context window in samples.
            model_name (str): Hugging Face id of the MOMENT checkpoint.
        """
        super().__init__()
        from momentfm import MOMENTPipeline  # pylint: disable=import-outside-toplevel
        self.context_length = context_length
        self.pipeline = MOMENTPipeline.from_pretrained(
            model_name, model_kwargs={"task_name": "reconstruction"})
        self.pipeline.init()
        for name, parameter in self.pipeline.named_parameters():
            parameter.requires_grad_(finetune or "head" in name)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition
        accel = inputs[:, 0, :]
        batch, length = accel.shape
        chunk = self.context_length
        pad = (-length) % chunk
        if pad:
            accel = torch.nn.functional.pad(accel, (0, pad))
        chunks = accel.view(batch, -1, chunk).reshape(-1, 1, chunk)
        mask = torch.ones(chunks.shape[0], chunk, device=accel.device)
        output = self.pipeline(x_enc=chunks, input_mask=mask).reconstruction
        return output.view(batch, 1, -1)[:, :, :length]


class TimesFMEncoderModel(nn.Module):
    """TimesFM 2.5 as a frozen encoder with a trained patch head.

    The third pretrained baseline, next to Chronos and MOMENT: a 200M
    decoder-only forecaster whose patch embeddings are reused as features.
    The series is split into context windows, each window is normalized and
    patched (32 samples per patch), the 20-layer stack produces one
    embedding per patch and a small convolutional head writes the moment
    samples of that patch back, with the window scale as an extra channel.
    """

    def __init__(self,
                 input_channels: int = 4,
                 context_length: int = 1024,
                 model_name: str = "google/timesfm-2.5-200m-pytorch",
                 finetune: bool = False):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels (only the first,
              the acceleration, is encoded).
            context_length (int): Context window in samples (multiple of 32).
            model_name (str): Hugging Face id of the TimesFM checkpoint.
            finetune (bool): Train the encoder as well as the head.
        """
        super().__init__()
        # pylint: disable=import-outside-toplevel
        from timesfm import timesfm_2p5_torch
        wrapper = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
            model_name, torch_compile=False)
        self.encoder = wrapper.model
        self.finetune = finetune
        self.patch = self.encoder.p
        self.context_length = context_length - context_length % self.patch
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(finetune)
        self.head = nn.Sequential(
            nn.Conv1d(self.encoder.md + 1, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, self.patch, 1),
        )

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition
        accel = inputs[:, 0, :]
        length = accel.shape[-1]
        chunk = self.context_length
        padding = (-length) % chunk
        if padding:
            accel = nn.functional.pad(accel, (0, padding))
        windows = accel.reshape(-1, chunk)
        scale = windows.std(dim=-1, keepdim=True) + 1e-6
        patches = (windows / scale).reshape(-1, chunk // self.patch, self.patch)
        with torch.set_grad_enabled(self.finetune):
            embeddings = self.encoder(
                patches, torch.zeros_like(patches, dtype=torch.bool))[0][1]
        tokens = embeddings.transpose(1, 2)
        scale_row = scale[:, :, None].expand(-1, 1, tokens.shape[-1])
        moment = self.head(torch.cat([tokens, scale_row], dim=1))
        moment = moment.transpose(1, 2).reshape(inputs.shape[0], 1, -1)
        return moment[..., :length]


class ProbabilisticTCNModel(nn.Module):
    """TCN with a heteroscedastic head: predicts mean and log-variance.

    Trained with the Gaussian negative log-likelihood, it reports how much
    of the target it believes it can explain. This is the right output for
    partially observable tasks such as the excitation-driven surrogate,
    where part of the response is irreducible given the inputs.
    """

    predicts_variance = True

    def __init__(self,
                 input_channels: int = 4,
                 hidden_channels: int = 64,
                 kernel_size: int = 5,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
                 dropout: float = 0.0):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Width of the residual blocks.
            kernel_size (int): Convolution kernel size.
            dilations (Sequence[int]): Dilation of each residual block.
            dropout (float): Dropout inside each block (0 = none).
        """
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden_channels, kernel_size, dilation, dropout)
            for dilation in dilations
        ])
        self.output_proj = nn.Conv1d(hidden_channels, 2, 1)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps inputs to (batch, 2, length): mean and log-variance."""
        del condition
        hidden = self.input_proj(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.output_proj(hidden)
        return torch.cat([output[:, :1, :], output[:, 1:, :].clamp(-8.0, 8.0)],
                         dim=1)


class TCNDamageModel(nn.Module):
    """End-to-end damage prediction: TCN encoder + pooling + scalar head.

    Predicts the (normalized) log10 fatigue damage of the full window
    directly from the acceleration and operating-state channels, skipping
    the moment reconstruction. Used to test whether the physically
    grounded intermediate (moment series + exact rainflow) helps or hurts
    against direct end-to-end regression.
    """

    def __init__(self,
                 input_channels: int = 4,
                 hidden_channels: int = 64,
                 kernel_size: int = 5,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
                 head_hidden: int = 64):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Width of the residual blocks.
            kernel_size (int): Convolution kernel size.
            dilations (Sequence[int]): Dilation of each residual block.
            head_hidden (int): Hidden width of the scalar head.
        """
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden_channels, kernel_size, dilation)
            for dilation in dilations
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch,) predictions."""
        del condition  # The state enters through the temporal channels.
        hidden = self.input_proj(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = hidden.mean(dim=-1)
        return self.head(pooled)[:, 0]


class SequenceScalarModel(nn.Module):
    """End-to-end damage prediction with any sequence backbone.

    The backbone writes an internal one-channel series, never supervised,
    and the head reads three pooled statistics of it (log standard
    deviation, log peak amplitude and the log spectral damage proxy
    sum|A|^m f) to predict the normalized log10 damage. It exists so the
    end-to-end task can be run with architectures other than the TCN, and
    so answers whether its behaviour is a property of the task or of that
    one model. The statistics are the quantities fatigue actually depends
    on, which keeps the head small and the comparison about the backbone.
    """

    def __init__(self,
                 backbone: str,
                 num_samples: int,
                 input_channels: int = 4,
                 condition_dim: int = 3,
                 damage_m: float = 4.0,
                 head_hidden: int = 64):
        """Initializes the model.

        Args:
            backbone (str): Name of the sequence model to wrap.
            num_samples (int): Sequence length passed to the backbone.
            input_channels (int): Number of input channels.
            condition_dim (int): Number of condition scalars.
            damage_m (float): Amplitude exponent of the damage proxy.
            head_hidden (int): Hidden width of the scalar head.
        """
        super().__init__()
        self.damage_m = damage_m
        self.backbone = build_model(backbone,
                                    num_samples=num_samples,
                                    input_channels=input_channels,
                                    condition_dim=condition_dim)
        self.head = nn.Sequential(
            nn.Linear(3, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch,) predictions."""
        series = self.backbone(inputs, condition)[:, 0, :]
        amplitudes = torch.abs(torch.fft.rfft(series,
                                              dim=-1)) * 2.0 / series.shape[-1]
        frequencies = torch.arange(amplitudes.shape[-1],
                                   device=series.device,
                                   dtype=series.dtype)
        proxy = torch.sum(amplitudes**self.damage_m * frequencies, dim=-1)
        features = torch.stack([
            torch.log(series.std(dim=-1) + 1e-12),
            torch.log(series.abs().amax(dim=-1) + 1e-12),
            torch.log(proxy + 1e-12),
        ],
                               dim=-1)
        return self.head(features)[:, 0]


class NaiveGainModel(nn.Module):
    """Trivial floor: one learned scalar gain, no conditioning.

    Reconstructs the moment as a single scalar times the acceleration. It
    anchors the bottom of the leaderboard: any model that cannot beat it is
    not using the operating state or the frequency structure at all.
    """

    def __init__(self, input_channels: int = 4):
        """Initializes the model.

        Args:
            input_channels (int): Unused, kept for a uniform constructor.
        """
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition
        return self.gain * inputs[:, :1, :]


class HybridTCNModel(nn.Module):
    """Physics reconstruction plus a bounded temporal correction.

    The frequency-domain hybrid corrects the physics gains with a
    scalar-conditioned hypernetwork; this variant keeps the same bounded
    principle but lets a TCN, the model that transfers best, write the
    correction in the time domain:

        m = m_phys + bound * std(m_phys) * tanh(TCN(inputs))

    so the correction can never exceed `bound` times the amplitude of the
    physics reconstruction.
    """

    needs_physics_gain = True

    def __init__(self, input_channels: int = 4, bound: float = 0.5):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            bound (float): Maximum correction, in units of std(m_phys).
        """
        super().__init__()
        self.bound = bound
        self.residual = TCNModel(input_channels=input_channels)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor,
                physics_gain: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        accel = inputs[:, 0, :]
        spectrum = torch.fft.rfft(accel, dim=-1)
        moment_phys = torch.fft.irfft(spectrum *
                                      physics_gain.to(spectrum.dtype),
                                      n=accel.shape[-1],
                                      dim=-1)
        scale = moment_phys.std(dim=-1, keepdim=True)
        correction = torch.tanh(self.residual(inputs, condition)[:, 0, :])
        return (moment_phys + self.bound * scale * correction)[:, None, :]


class DLinearModel(nn.Module):
    """Decomposition linear baseline (DLinear), length-agnostic.

    Splits the acceleration into a moving-average trend and the remainder
    and maps each with its own depthwise convolution over a fixed context.
    It is the standard sanity check of the forecasting literature: a linear
    map that repeatedly matches deep architectures.
    """

    def __init__(self,
                 input_channels: int = 4,
                 kernel_size: int = 101,
                 context: int = 257):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            kernel_size (int): Moving-average window of the decomposition.
            context (int): Receptive field of each linear branch (odd).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.trend = nn.Conv1d(input_channels, 1, context, padding=context // 2)
        self.seasonal = nn.Conv1d(input_channels,
                                  1,
                                  context,
                                  padding=context // 2)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition
        pad = self.kernel_size // 2
        padded = nn.functional.pad(inputs, (pad, pad), mode="replicate")
        trend = nn.functional.avg_pool1d(padded, self.kernel_size, stride=1)
        return self.trend(trend) + self.seasonal(inputs - trend)


class _SelectiveSSMLayer(nn.Module):
    """Selective state-space layer (Mamba's S6) with a chunked scan.

    The recurrence h_t = exp(dt_t * A) h_{t-1} + dt_t * B_t * x_t is solved
    without the fused CUDA kernel: the sequence is split into chunks, each
    chunk is scanned sequentially from a zero state, and the chunk initial
    states are propagated with the cumulative decay. The cost is
    O(chunk + length / chunk) kernel launches instead of O(length).
    """

    def __init__(self,
                 channels: int = 64,
                 state_dim: int = 4,
                 expand: int = 2,
                 conv_kernel: int = 4,
                 chunk: int = 64):
        """Initializes the layer.

        Args:
            channels (int): Width of the residual stream.
            state_dim (int): Number of states per channel.
            expand (int): Expansion factor of the inner branch.
            conv_kernel (int): Depthwise causal convolution kernel.
            chunk (int): Chunk length of the scan.
        """
        super().__init__()
        inner = expand * channels
        self.state_dim = state_dim
        self.inner = inner
        self.chunk = chunk
        self.conv_kernel = conv_kernel
        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv1d(channels, 2 * inner, 1)
        self.conv = nn.Conv1d(inner,
                              inner,
                              conv_kernel,
                              padding=conv_kernel - 1,
                              groups=inner)
        self.to_delta = nn.Conv1d(inner, inner, 1)
        self.to_state = nn.Conv1d(inner, 2 * state_dim, 1)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1,
                                   dtype=torch.float32)).repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Conv1d(inner, channels, 1)

    def _scan(self, decay: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Scans h_t = decay_t * h_{t-1} + source_t over the last axis."""
        length = source.shape[-1]
        chunk = min(self.chunk, length)
        padding = (-length) % chunk
        if padding:
            decay = nn.functional.pad(decay, (0, padding), value=1.0)
            source = nn.functional.pad(source, (0, padding))
        shape = source.shape[:3] + (decay.shape[-1] // chunk, chunk)
        decay = decay.reshape(shape)
        source = source.reshape(shape)
        state = torch.zeros_like(source[..., 0])
        local = []
        for index in range(chunk):
            state = decay[..., index] * state + source[..., index]
            local.append(state)
        local = torch.stack(local, dim=-1)
        cumulative = torch.cumprod(decay, dim=-1)
        carry = torch.zeros_like(local[..., 0, 0])
        initial = []
        for index in range(shape[3]):
            initial.append(carry)
            carry = (cumulative[..., index, -1] * carry + local[..., index, -1])
        output = local + cumulative * torch.stack(initial, dim=-1)[..., None]
        return output.reshape(shape[:3] + (-1,))[..., :length]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the layer to (batch, channels, length) inputs."""
        length = inputs.shape[-1]
        projected = self.in_proj(self.norm(inputs))
        hidden, gate = projected[:, :self.inner], projected[:, self.inner:]
        hidden = nn.functional.silu(self.conv(hidden)[..., :length])
        delta = nn.functional.softplus(self.to_delta(hidden))
        state_params = self.to_state(hidden)
        b_matrix = state_params[:, :self.state_dim, :]
        c_matrix = state_params[:, self.state_dim:, :]
        a_matrix = -torch.exp(self.a_log)
        decay = torch.exp(delta[:, :, None, :] * a_matrix[None, :, :, None])
        source = (delta[:, :, None, :] * b_matrix[:, None, :, :] *
                  hidden[:, :, None, :])
        states = self._scan(decay, source)
        output = torch.sum(states * c_matrix[:, None, :, :], dim=2)
        output = output + self.skip[None, :, None] * hidden
        return inputs + self.out_proj(output * nn.functional.silu(gate))


class MambaModel(nn.Module):
    """Selective state-space model (Mamba/S6), length-agnostic.

    Complements the S4D baseline: the state transition and the input and
    output projections are functions of the signal itself, so the model can
    change its effective time constants with the operating state instead of
    holding them fixed.
    """

    def __init__(self,
                 input_channels: int = 4,
                 hidden_channels: int = 64,
                 state_dim: int = 4,
                 num_layers: int = 4):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Width of the residual stream.
            state_dim (int): Number of states per channel.
            num_layers (int): Number of selective state-space layers.
        """
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, 1)
        self.layers = nn.ModuleList([
            _SelectiveSSMLayer(channels=hidden_channels, state_dim=state_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The state enters through the temporal channels.
        hidden = self.input_proj(inputs)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output_proj(hidden)


class _TimesBlock(nn.Module):
    """Period-folded 2D convolution block (TimesNet)."""

    def __init__(self,
                 channels: int,
                 num_periods: int = 3,
                 kernels: Sequence[int] = (3, 5)):
        super().__init__()
        self.num_periods = num_periods
        self.inception = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel, padding=kernel // 2)
            for kernel in kernels
        ])
        self.activation = nn.GELU()
        self.project = nn.Conv2d(channels, channels, 1)

    def _periods(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns the dominant periods and their amplitude weights."""
        amplitude = torch.abs(torch.fft.rfft(inputs.mean(dim=1), dim=-1))
        amplitude = amplitude.mean(dim=0)
        amplitude[0] = 0.0
        top = min(self.num_periods, amplitude.shape[0] - 1)
        values, indices = torch.topk(amplitude, top)
        periods = torch.clamp(inputs.shape[-1] // indices.clamp(min=1),
                              min=2,
                              max=inputs.shape[-1])
        return periods, torch.softmax(values, dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the block to (batch, channels, length) inputs."""
        length = inputs.shape[-1]
        periods, weights = self._periods(inputs)
        output = torch.zeros_like(inputs)
        for index, period in enumerate(periods.tolist()):
            padding = (-length) % period
            folded = nn.functional.pad(inputs, (0, padding), mode="replicate")
            folded = folded.reshape(inputs.shape[0], inputs.shape[1], -1,
                                    period)
            hidden = sum(conv(folded) for conv in self.inception)
            hidden = self.project(self.activation(hidden))
            hidden = hidden.reshape(inputs.shape[0], inputs.shape[1],
                                    -1)[..., :length]
            output = output + weights[index] * hidden
        return inputs + output


class TimesNetModel(nn.Module):
    """TimesNet: folds the series at its dominant periods and convolves.

    The rotor harmonics (1P, 3P, 6P) and the wave period make the signal
    quasi-periodic, so folding the series into a 2D image whose rows are
    successive cycles puts samples with the same phase side by side. It is
    the strongest published inductive bias for periodic time series and the
    natural competitor of the band-based physics baseline.
    """

    def __init__(self,
                 input_channels: int = 4,
                 hidden_channels: int = 48,
                 num_blocks: int = 2,
                 num_periods: int = 3):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Width of the residual stream.
            num_blocks (int): Number of period-folded blocks.
            num_periods (int): Dominant periods kept per block.
        """
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            _TimesBlock(hidden_channels, num_periods=num_periods)
            for _ in range(num_blocks)
        ])
        self.norms = nn.ModuleList(
            [nn.GroupNorm(1, hidden_channels) for _ in range(num_blocks)])
        self.output_proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The state enters through the temporal channels.
        hidden = self.input_proj(inputs)
        for block, norm in zip(self.blocks, self.norms):
            hidden = norm(block(hidden))
        return self.output_proj(hidden)


class FITSModel(nn.Module):
    """Complex linear map between low-frequency bands (FITS).

    Keeps the bins below a cutoff of the rFFT of every input channel and
    maps them to the output bins with a single complex matrix; everything
    above the cutoff is dropped. Unlike `SpectralGainModel` the map is not
    diagonal (bins can feed each other) and not conditioned, which isolates
    how much of the task is a fixed linear filter. The sequence length is
    fixed (`num_samples`).
    """

    def __init__(self,
                 num_samples: int,
                 input_channels: int = 4,
                 cutoff_bins: int = 768):
        """Initializes the model.

        Args:
            num_samples (int): Fixed input sequence length.
            input_channels (int): Number of input channels.
            cutoff_bins (int): Number of low-frequency bins kept.
        """
        super().__init__()
        self.num_bins = num_samples // 2 + 1
        self.cutoff = min(cutoff_bins, self.num_bins)
        scale = 1.0 / (input_channels * self.cutoff)**0.5
        self.weight = nn.Parameter(
            scale * torch.randn(input_channels * self.cutoff, self.cutoff, 2))

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The state enters through the temporal channels.
        length = inputs.shape[-1]
        spectrum = torch.fft.rfft(inputs, dim=-1)[..., :self.cutoff]
        spectrum = spectrum.reshape(spectrum.shape[0], -1)
        weight = torch.complex(self.weight[..., 0], self.weight[..., 1])
        low = spectrum @ weight
        full = torch.zeros(low.shape[0],
                           length // 2 + 1,
                           dtype=low.dtype,
                           device=low.device)
        full[:, :self.cutoff] = low
        return torch.fft.irfft(full, n=length, dim=-1)[:, None, :]


class ITransformerModel(nn.Module):
    """Inverted transformer: attention across channels, not time.

    Every input channel becomes a single token holding its whole series, so
    attention models the relations between acceleration, wind, wave and
    rotor state directly. A learned query token is decoded into the moment
    series. The sequence length is fixed (`num_samples`).
    """

    def __init__(self,
                 num_samples: int,
                 input_channels: int = 4,
                 embed_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initializes the model.

        Args:
            num_samples (int): Fixed input sequence length.
            input_channels (int): Number of input channels.
            embed_dim (int): Token embedding width.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Attention heads per layer.
            dropout (float): Dropout of the encoder layers (0.1, the torch
              default, as published).
        """
        super().__init__()
        self.num_samples = num_samples
        self.embed = nn.Linear(num_samples, embed_dim)
        self.variate = nn.Parameter(0.02 *
                                    torch.randn(1, input_channels, embed_dim))
        self.query = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim,
                                           num_heads,
                                           dim_feedforward=4 * embed_dim,
                                           dropout=dropout,
                                           batch_first=True,
                                           activation="gelu",
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(embed_dim, num_samples)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The state enters through the temporal channels.
        tokens = self.embed(inputs) + self.variate
        query = self.query.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([query, tokens], dim=1))
        return self.head(encoded[:, 0])[:, None, :]


class _UNetBlock(nn.Module):
    """Two convolutions with GELU activations at a fixed resolution."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GELU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the block to (batch, channels, length) inputs."""
        return self.body(inputs)


class UNetModel(nn.Module):
    """Multi-rate convolutional encoder-decoder, length-agnostic.

    Downsamples the signal four times and reconstructs it with skip
    connections, so each stage works at one time scale (the wave band, the
    tower modes, the rotor harmonics). It is the convolutional counterpart
    of the multi-rate stacks used by N-HiTS.
    """

    def __init__(self,
                 input_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 kernel_size: int = 5):
        """Initializes the model.

        Args:
            input_channels (int): Number of input channels.
            base_channels (int): Width of the first stage.
            num_stages (int): Number of downsampling stages.
            kernel_size (int): Kernel of every convolution (odd).
        """
        super().__init__()
        self.num_stages = num_stages
        widths = [base_channels * 2**stage for stage in range(num_stages + 1)]
        self.encoders = nn.ModuleList([
            _UNetBlock(input_channels if stage == 0 else widths[stage - 1],
                       widths[stage], kernel_size)
            for stage in range(num_stages)
        ])
        self.bottleneck = _UNetBlock(widths[num_stages - 1], widths[num_stages],
                                     kernel_size)
        self.decoders = nn.ModuleList([
            _UNetBlock(widths[stage + 1] + widths[stage], widths[stage],
                       kernel_size) for stage in range(num_stages)
        ])
        self.output_proj = nn.Conv1d(base_channels, 1, 1)

    def forward(self, inputs: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Maps (batch, channels, length) inputs to (batch, 1, length)."""
        del condition  # The state enters through the temporal channels.
        length = inputs.shape[-1]
        stride = 2**self.num_stages
        padding = (-length) % stride
        hidden = nn.functional.pad(inputs, (0, padding), mode="replicate")
        skips = []
        for encoder in self.encoders:
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = nn.functional.avg_pool1d(hidden, 2)
        hidden = self.bottleneck(hidden)
        for stage in reversed(range(self.num_stages)):
            hidden = nn.functional.interpolate(hidden,
                                               size=skips[stage].shape[-1],
                                               mode="linear",
                                               align_corners=False)
            hidden = self.decoders[stage](torch.cat([hidden, skips[stage]],
                                                    dim=1))
        return self.output_proj(hidden)[..., :length]


# Models whose parameters depend on the series length: they train and are
# scored on the full window instead of random crops.
LENGTH_FIXED_MODELS = ("spectral", "hybrid", "hybrid_tcn", "fits",
                       "itransformer")
# Models that read the first input channel as the tower-top acceleration.
ACCEL_FIRST_MODELS = ("naive", "spectral", "hybrid", "hybrid_tcn", "chronos",
                      "moment", "moment_ft", "timesfm", "timesfm_ft")


def _tcn_kwargs(model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """TCN kwargs with `num_levels` turned into the dilations 2**i."""
    kwargs = dict(model_kwargs)
    if "num_levels" in kwargs:
        kwargs["dilations"] = tuple(
            2**i for i in range(kwargs.pop("num_levels")))
    return kwargs


def build_model(name: str,
                num_samples: int,
                input_channels: int,
                condition_dim: int,
                condition_bound: float = 0.5,
                **model_kwargs) -> nn.Module:
    """Builds a model by name ('spectral', 'hybrid', 'tcn', 'lstm',
    'transformer', 'mamba', 'fits', 'itransformer', 'unet', ...).

    `model_kwargs` go to the model constructor (the architecture knobs of
    the validation-tuned track); without them every model is the published
    one. TCN and Prob-TCN also take `num_levels` (dilations 2**i).
    """
    kwargs = model_kwargs
    if name == "spectral":
        return SpectralGainModel(num_samples=num_samples,
                                 condition_dim=condition_dim,
                                 **kwargs)
    if name == "hybrid":
        return HybridGainModel(num_samples=num_samples,
                               condition_bound=condition_bound,
                               condition_dim=condition_dim,
                               **kwargs)
    if name == "tcn":
        return TCNModel(input_channels=input_channels, **_tcn_kwargs(kwargs))
    if name == "dlinear":
        return DLinearModel(input_channels=input_channels, **kwargs)
    if name == "naive":
        return NaiveGainModel(input_channels=input_channels, **kwargs)
    if name == "hybrid_tcn":
        return HybridTCNModel(input_channels=input_channels,
                              bound=condition_bound,
                              **kwargs)
    if name == "s4":
        return S4Model(input_channels=input_channels, **kwargs)
    if name == "mamba":
        return MambaModel(input_channels=input_channels, **kwargs)
    if name == "unet":
        return UNetModel(input_channels=input_channels, **kwargs)
    if name == "timesnet":
        return TimesNetModel(input_channels=input_channels, **kwargs)
    if name == "fits":
        return FITSModel(num_samples=num_samples,
                         input_channels=input_channels,
                         **kwargs)
    if name == "itransformer":
        return ITransformerModel(num_samples=num_samples,
                                 input_channels=input_channels,
                                 **kwargs)
    if name == "fno":
        return FNOModel(input_channels=input_channels, **kwargs)
    if name == "moment_ft":
        return MomentModel(input_channels=input_channels,
                           finetune=True,
                           **kwargs)
    if name == "timesfm":
        return TimesFMEncoderModel(input_channels=input_channels, **kwargs)
    if name == "timesfm_ft":
        return TimesFMEncoderModel(input_channels=input_channels,
                                   finetune=True,
                                   **kwargs)
    if name == "chronos":
        return ChronosEncoderModel(input_channels=input_channels, **kwargs)
    if name == "moment":
        return MomentModel(input_channels=input_channels, **kwargs)
    if name == "prob_tcn":
        return ProbabilisticTCNModel(input_channels=input_channels,
                                     **_tcn_kwargs(kwargs))
    if name == "lstm":
        return LSTMModel(input_channels=input_channels, **kwargs)
    if name == "transformer":
        return PatchTransformerModel(input_channels=input_channels, **kwargs)
    raise ValueError(f"Unknown model: '{name}'.")


def count_parameters(model: nn.Module, trainable: bool = True) -> int:
    """Number of (trainable) parameters of a model."""
    return sum(p.numel()
               for p in model.parameters()
               if p.requires_grad or not trainable)


def _typed(value: str) -> Any:
    """'true'/'false' -> bool, then int, then float, else the string."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_model_kwargs(text: Optional[str]) -> Dict[str, Any]:
    """Parses 'k=v,k=v' (the --model_kwargs flag) into typed kwargs.

    Args:
        text (str, optional): Comma-separated key=value pairs; empty or None
          gives no kwargs.

    Returns:
        dict: Keyword arguments with bool, int, float or str values.

    Raises:
        ValueError: On a pair without '=', an empty key or a repeated key.
    """
    kwargs = {}
    for pair in filter(None, (text or "").split(",")):
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key or key in kwargs:
            raise ValueError(f"Bad --model_kwargs entry '{pair}' in '{text}'.")
        kwargs[key] = _typed(value.strip())
    return kwargs
