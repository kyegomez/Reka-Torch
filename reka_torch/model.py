import torch
from torch import nn
from zeta.nn.attention import Attention
from zeta.nn import FeedForward
from zeta.nn import audio_to_text, img_to_text, video_to_text


class RekaTransformerBlock(nn.Module):
    """
    RekaTransformerBlock is a module that represents a single transformer block in the RekaTransformer model.

    Args:
        dim (int): The input dimension of the transformer block.
        depth (int): The number of layers in the transformer block.
        dim_head (int, optional): The dimension of each head in the multi-head attention mechanism. Defaults to 64.
        heads (int, optional): The number of attention heads. Defaults to 8.
        ff_mult (int, optional): The multiplier for the hidden dimension in the feed-forward network. Defaults to 4.
        ff_dropout (float, optional): The dropout probability for the feed-forward network. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.ff_dropout = ff_dropout

        # Attention Layers
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            causal=True,
            qk_norm=True,
            kv_heads=4,
        )

        # Feed Forward Layers
        self.ffn = FeedForward(
            dim,
            dim,
            mult=ff_mult,
            post_act_ln=True,
            swish=True,
        )

    def forward(self, x):
        """
        Forward pass of the RekaTransformerBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        skip = x
        x, _ = self.attn(x)
        # print(x.shape, skip.shape)
        attended = x + skip

        x = self.ffn(attended)
        x += skip
        return x


class RekaTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.ff_dropout = ff_dropout

        self.blocks = nn.ModuleList(
            [
                RekaTransformerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Reka(nn.Module):
    """
    Reka model for multimodal fusion.

    Args:
        dim (int): Dimension of the input features.
        depth (int, optional): Number of transformer layers. Defaults to 6.
        dim_head (int, optional): Dimension of each head in the transformer. Defaults to 64.
        heads (int, optional): Number of attention heads in the transformer. Defaults to 8.
        ff_mult (int, optional): Multiplier for the feed-forward layer dimension. Defaults to 4.
        ff_dropout (float, optional): Dropout probability for the feed-forward layer. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
        post_modal_transform_norm: bool = True,
        post_fusion_norm: bool = True,
        vocab_size: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.ff_dropout = ff_dropout
        self.post_modal_transform_norm = post_modal_transform_norm
        self.post_fusion_norm = post_fusion_norm

        self.transformer = RekaTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
        )

        # Post fusion norm
        if self.post_fusion_norm:
            self.psf_norm = nn.LayerNorm(dim)

        if self.post_modal_transform_norm:
            self.pmt_norm = nn.LayerNorm(dim)

        # Embedder
        self.token = nn.Embedding(vocab_size, dim)

    def forward(self, text, img, audio, video):
        """
        Forward pass of the Reka model.

        Args:
            text: Not used in this implementation.
            img (torch.Tensor): Input image tensor.
            audio (torch.Tensor): Input audio tensor.
            video (torch.Tensor): Input video tensor.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        logits = self.token(text)

        assert (
            (img is not None and audio is not None)
            or (img is not None and video is not None)
            or (audio is not None and video is not None)
        ), (
            "At least two of the inputs (img, audio, video) must be"
            " provided."
        )

        if img is not None:
            # Image dimensions
            img_b, img_c, img_h, img_w = img.shape

            # img = img_to_text(img, self.patches, self.patch_size, self.dim, True)
            img = img_to_text(img, self.max_seq_len, self.dim, True)

            if self.post_modal_transform_norm:
                img = self.pmt_norm(img)

        if audio is not None:
            # Audio dimensions
            audio_b, audio_seq_len = audio.shape

            audio = audio_to_text(
                audio, self.max_seq_len, self.dim, True
            )

            if self.post_modal_transform_norm:
                audio = self.pmt_norm(audio)

        if video is not None:
            # Video dimensions
            video_b, video_c, video_f, video_h, video_w = video.shape

            video = video_to_text(
                video, self.max_seq_len, self.dim, True
            )

        # Fuse layers
        if img is not None and audio is not None:
            fused = torch.cat((img, audio), dim=1)
        elif img is not None and video is not None:
            fused = torch.cat((img, video), dim=1)
        elif audio is not None and video is not None:
            fused = torch.cat((audio, video), dim=1)

        # Post fusion layernorm for stability.
        if self.post_fusion_norm:
            fused = self.psf_norm(fused)

        # Fuse
        fused = torch.cat((logits, fused), dim=1)

        # Transformer
        x = self.transformer(fused)

        return x


text = torch.randint(0, 10000, (2, 512))

img = torch.randn(2, 3, 224, 224)

audio = torch.randn(2, 1000)

video = torch.randn(2, 3, 16, 224, 224)

model = Reka(512)

out = model(text, img, audio, video)
print(out.shape)
