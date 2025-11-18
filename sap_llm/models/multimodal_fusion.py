"""
Advanced Multi-Modal Fusion Layer for Vision + Language + Audio + Video Integration.

Combines multiple modalities using advanced fusion techniques:
- Vision: Image features from document pages
- Text: OCR and language features
- Audio: Speech-to-text transcriptions
- Video: Temporal keyframe features

Fusion Techniques:
- Cross-attention mechanism (32 heads)
- Intelligent gating (modality confidence weighting)
- Positional encoding for spatial relationships
- Temporal encoding for video sequences
- Cross-modal consistency checks
- Learnable fusion weights
- Attention visualization for interpretability

Target Metrics:
- Multi-modal fusion accuracy: ≥95%
- Fusion accuracy improvement: +5% vs simple concatenation
- Latency overhead: <50ms for vision+text, <100ms for all modalities
- Interpretability: Attention weights visualizable
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for spatial relationships.

    Encodes both row and column positions to capture document layout.
    """

    def __init__(self, d_model: int, max_h: int = 100, max_w: int = 100):
        """
        Initialize 2D positional encoding.

        Args:
            d_model: Embedding dimension
            max_h: Maximum height (rows)
            max_w: Maximum width (columns)
        """
        super().__init__()

        self.d_model = d_model

        # Create positional encodings for height and width
        pe_h = torch.zeros(max_h, d_model // 2)
        pe_w = torch.zeros(max_w, d_model // 2)

        position_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        position_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() *
            (-math.log(10000.0) / (d_model // 2))
        )

        # Sine and cosine for height
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)

        # Sine and cosine for width
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)

        # Register as buffers (not parameters)
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)

    def forward(self, x: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding based on bounding boxes.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            boxes: Bounding boxes [batch_size, seq_len, 4] (normalized 0-1)

        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, _ = x.shape

        # Extract center positions from boxes
        # boxes format: [x0, y0, x1, y1]
        center_x = ((boxes[:, :, 0] + boxes[:, :, 2]) / 2 * 99).long()
        center_y = ((boxes[:, :, 1] + boxes[:, :, 3]) / 2 * 99).long()

        # Clamp to valid range
        center_x = torch.clamp(center_x, 0, 99)
        center_y = torch.clamp(center_y, 0, 99)

        # Get positional encodings
        pos_enc = torch.zeros_like(x)

        for i in range(batch_size):
            for j in range(seq_len):
                h_enc = self.pe_h[center_y[i, j]]
                w_enc = self.pe_w[center_x[i, j]]
                pos_enc[i, j] = torch.cat([h_enc, w_enc], dim=0)

        return x + pos_enc


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism between vision and text features.

    Allows vision features to attend to text features and vice versa.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 32,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization
        self.attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention.

        Args:
            query: Query features [batch_size, q_len, dim]
            key: Key features [batch_size, k_len, dim]
            value: Value features [batch_size, v_len, dim]
            mask: Attention mask [batch_size, q_len, k_len]

        Returns:
            Tuple of (attended features, attention weights)
        """
        batch_size, q_len, _ = query.shape
        _, k_len, _ = key.shape

        # Project to Q, K, V
        Q = self.q_proj(query)  # [batch_size, q_len, dim]
        K = self.k_proj(key)    # [batch_size, k_len, dim]
        V = self.v_proj(value)  # [batch_size, k_len, dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Store for visualization
        self.attention_weights = attn_weights.detach()

        # Weighted sum
        out = torch.matmul(attn_weights, V)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.dim)
        out = self.out_proj(out)

        return out, attn_weights


class GatingMechanism(nn.Module):
    """
    Intelligent gating to decide when to trust vision vs text.

    Learns to dynamically weight vision and text contributions based on:
    - Feature quality/confidence
    - Document type
    - Spatial context
    """

    def __init__(self, dim: int):
        """
        Initialize gating mechanism.

        Args:
            dim: Feature dimension
        """
        super().__init__()

        # Gate network: learns to produce gate values in [0, 1]
        self.gate_network = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),  # Gate value between 0 and 1
        )

        # Feature transformation
        self.vision_transform = nn.Linear(dim, dim)
        self.text_transform = nn.Linear(dim, dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated fusion of vision and text features.

        Args:
            vision_features: Vision features [batch_size, seq_len, dim]
            text_features: Text features [batch_size, seq_len, dim]

        Returns:
            Tuple of (fused features, gate values)
        """
        # Concatenate for gate computation
        concat_features = torch.cat([vision_features, text_features], dim=-1)

        # Compute gate values (0 = all text, 1 = all vision)
        gate = self.gate_network(concat_features)  # [batch_size, seq_len, 1]

        # Transform features
        vision_transformed = self.vision_transform(vision_features)
        text_transformed = self.text_transform(text_features)

        # Gated fusion
        fused = gate * vision_transformed + (1 - gate) * text_transformed

        return fused, gate


class MultiModalFusionLayer(nn.Module):
    """
    Advanced Multi-Modal Fusion Layer.

    Combines vision and language features using:
    1. Cross-attention (vision ← text, text ← vision)
    2. Intelligent gating mechanism
    3. 2D positional encoding
    4. Learnable fusion weights
    5. Residual connections

    Architecture:
    1. Add 2D positional encoding to vision features
    2. Cross-attention: vision attends to text
    3. Cross-attention: text attends to vision
    4. Intelligent gating between modalities
    5. Residual connection + LayerNorm

    Target Metrics:
    - Fusion accuracy: +5% vs simple concatenation
    - Latency overhead: <50ms
    - Interpretability: Attention weights stored for visualization
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 768,
        num_heads: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        enable_positional_encoding: bool = True,
    ):
        """
        Initialize multi-modal fusion layer.

        Args:
            vision_dim: Vision feature dimension
            text_dim: Text feature dimension
            fusion_dim: Fused feature dimension
            num_heads: Number of attention heads
            num_layers: Number of fusion layers
            dropout: Dropout rate
            enable_positional_encoding: Enable 2D positional encoding
        """
        super().__init__()

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.enable_positional_encoding = enable_positional_encoding

        logger.info(
            f"Initializing MultiModalFusionLayer: "
            f"vision_dim={vision_dim}, text_dim={text_dim}, "
            f"fusion_dim={fusion_dim}, num_heads={num_heads}, "
            f"num_layers={num_layers}"
        )

        # Project vision and text to same dimension
        self.vision_projection = nn.Linear(vision_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)

        # 2D Positional encoding
        if enable_positional_encoding:
            self.positional_encoding = PositionalEncoding2D(fusion_dim)

        # Multi-layer fusion
        self.fusion_layers = nn.ModuleList([
            FusionBlock(
                dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)

        # Store attention maps for visualization
        self.attention_maps = []

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_boxes: Optional[torch.Tensor] = None,
        text_boxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse vision and text features.

        Args:
            vision_features: Vision features [batch_size, vision_len, vision_dim]
            text_features: Text features [batch_size, text_len, text_dim]
            vision_boxes: Vision bounding boxes [batch_size, vision_len, 4]
            text_boxes: Text bounding boxes [batch_size, text_len, 4]

        Returns:
            Dictionary containing:
            - fused_features: Fused features [batch_size, seq_len, fusion_dim]
            - vision_features: Enhanced vision features
            - text_features: Enhanced text features
            - attention_maps: List of attention weight tensors
            - gate_values: Gate values from gating mechanism
        """
        # Project to fusion dimension
        vision_proj = self.vision_projection(vision_features)
        text_proj = self.text_projection(text_features)

        # Add positional encoding
        if self.enable_positional_encoding:
            if vision_boxes is not None:
                vision_proj = self.positional_encoding(vision_proj, vision_boxes)
            if text_boxes is not None:
                text_proj = self.positional_encoding(text_proj, text_boxes)

        # Multi-layer fusion
        self.attention_maps = []

        for i, fusion_layer in enumerate(self.fusion_layers):
            vision_proj, text_proj, layer_attention = fusion_layer(
                vision_proj,
                text_proj,
            )
            self.attention_maps.append(layer_attention)

        # Combine vision and text (simple concatenation along sequence dimension)
        fused_features = torch.cat([vision_proj, text_proj], dim=1)

        # Output projection
        fused_features = self.output_projection(fused_features)

        return {
            "fused_features": fused_features,
            "vision_features": vision_proj,
            "text_features": text_proj,
            "attention_maps": self.attention_maps,
        }

    def visualize_attention(
        self,
        layer_idx: int = 0,
        head_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for visualization.

        Args:
            layer_idx: Layer index to visualize
            head_idx: Attention head index to visualize

        Returns:
            Attention weights tensor or None if not available
        """
        if not self.attention_maps:
            logger.warning("No attention maps available for visualization")
            return None

        if layer_idx >= len(self.attention_maps):
            logger.warning(f"Layer index {layer_idx} out of range")
            return None

        layer_attention = self.attention_maps[layer_idx]

        # Get specific attention map
        if "vision_to_text" in layer_attention:
            attn = layer_attention["vision_to_text"]
            if head_idx < attn.shape[1]:
                return attn[:, head_idx, :, :]  # [batch_size, q_len, k_len]

        return None


class FusionBlock(nn.Module):
    """
    Single fusion block with bidirectional cross-attention and gating.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 32,
        dropout: float = 0.1,
    ):
        """
        Initialize fusion block.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Cross-attention: vision ← text
        self.vision_to_text_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention: text ← vision
        self.text_to_vision_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Gating mechanism
        self.vision_gate = GatingMechanism(dim)
        self.text_gate = GatingMechanism(dim)

        # Layer normalization
        self.vision_norm1 = nn.LayerNorm(dim)
        self.vision_norm2 = nn.LayerNorm(dim)
        self.text_norm1 = nn.LayerNorm(dim)
        self.text_norm2 = nn.LayerNorm(dim)

        # Feed-forward networks
        self.vision_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.text_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through fusion block.

        Args:
            vision_features: Vision features [batch_size, vision_len, dim]
            text_features: Text features [batch_size, text_len, dim]

        Returns:
            Tuple of (enhanced vision, enhanced text, attention maps)
        """
        # Cross-attention: vision ← text
        vision_attn, vision_to_text_weights = self.vision_to_text_attn(
            query=vision_features,
            key=text_features,
            value=text_features,
        )

        # Cross-attention: text ← vision
        text_attn, text_to_vision_weights = self.text_to_vision_attn(
            query=text_features,
            key=vision_features,
            value=vision_features,
        )

        # Gating: combine original and attended features
        vision_gated, vision_gate_values = self.vision_gate(
            vision_features,
            vision_attn,
        )

        text_gated, text_gate_values = self.text_gate(
            text_features,
            text_attn,
        )

        # Residual connection + norm
        vision_features = self.vision_norm1(vision_features + vision_gated)
        text_features = self.text_norm1(text_features + text_gated)

        # Feed-forward + residual
        vision_features = self.vision_norm2(
            vision_features + self.vision_ffn(vision_features)
        )
        text_features = self.text_norm2(
            text_features + self.text_ffn(text_features)
        )

        # Store attention maps
        attention_maps = {
            "vision_to_text": vision_to_text_weights,
            "text_to_vision": text_to_vision_weights,
            "vision_gate": vision_gate_values,
            "text_gate": text_gate_values,
        }

        return vision_features, text_features, attention_maps


class TemporalEncoder(nn.Module):
    """
    Temporal encoding for video sequences.

    Encodes frame positions and time information for video keyframes.
    """

    def __init__(self, d_model: int, max_frames: int = 100):
        """
        Initialize temporal encoder.

        Args:
            d_model: Embedding dimension
            max_frames: Maximum number of frames
        """
        super().__init__()

        self.d_model = d_model

        # Create temporal positional encodings
        pe = torch.zeros(max_frames, d_model)
        position = torch.arange(0, max_frames, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Add temporal encoding.

        Args:
            x: Input tensor [batch_size, num_frames, d_model]
            frame_indices: Frame indices [batch_size, num_frames]

        Returns:
            Tensor with temporal encoding added
        """
        batch_size, num_frames, _ = x.shape

        # Add temporal encoding based on frame indices
        temporal_enc = torch.zeros_like(x)

        for i in range(batch_size):
            for j in range(num_frames):
                frame_idx = frame_indices[i, j].long()
                if frame_idx < len(self.pe):
                    temporal_enc[i, j] = self.pe[frame_idx]

        return x + temporal_enc


class AudioFeatureEncoder(nn.Module):
    """
    Encode audio features for fusion.

    Processes audio transcription embeddings and confidence scores.
    """

    def __init__(
        self,
        audio_dim: int = 768,
        output_dim: int = 768,
    ):
        """
        Initialize audio feature encoder.

        Args:
            audio_dim: Input audio feature dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        self.projection = nn.Linear(audio_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

        # Confidence embedding
        self.confidence_embedding = nn.Embedding(100, output_dim)

    def forward(
        self,
        audio_features: torch.Tensor,
        confidence_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode audio features.

        Args:
            audio_features: Audio features [batch_size, seq_len, audio_dim]
            confidence_scores: Confidence scores [batch_size, seq_len] (0-1)

        Returns:
            Encoded features [batch_size, seq_len, output_dim]
        """
        # Project audio features
        encoded = self.projection(audio_features)

        # Add confidence information if available
        if confidence_scores is not None:
            # Convert confidence to discrete bins (0-99)
            confidence_bins = (confidence_scores * 99).long()
            confidence_bins = torch.clamp(confidence_bins, 0, 99)

            # Add confidence embeddings
            conf_emb = self.confidence_embedding(confidence_bins)
            encoded = encoded + conf_emb

        # Normalize
        encoded = self.norm(encoded)

        return encoded


class VideoFeatureEncoder(nn.Module):
    """
    Encode video features for fusion.

    Processes keyframe features with temporal information.
    """

    def __init__(
        self,
        video_dim: int = 768,
        output_dim: int = 768,
        max_frames: int = 100,
    ):
        """
        Initialize video feature encoder.

        Args:
            video_dim: Input video feature dimension
            output_dim: Output feature dimension
            max_frames: Maximum number of keyframes
        """
        super().__init__()

        self.projection = nn.Linear(video_dim, output_dim)
        self.temporal_encoder = TemporalEncoder(output_dim, max_frames)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        video_features: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode video features.

        Args:
            video_features: Video features [batch_size, num_frames, video_dim]
            frame_indices: Frame indices [batch_size, num_frames]

        Returns:
            Encoded features [batch_size, num_frames, output_dim]
        """
        # Project video features
        encoded = self.projection(video_features)

        # Add temporal encoding
        if frame_indices is not None:
            encoded = self.temporal_encoder(encoded, frame_indices)

        # Normalize
        encoded = self.norm(encoded)

        return encoded


class CrossModalConsistencyChecker(nn.Module):
    """
    Check consistency across different modalities.

    Validates that information from different modalities is consistent.
    """

    def __init__(self, dim: int):
        """
        Initialize consistency checker.

        Args:
            dim: Feature dimension
        """
        super().__init__()

        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency score between two modalities.

        Args:
            features1: Features from modality 1 [batch_size, seq_len, dim]
            features2: Features from modality 2 [batch_size, seq_len, dim]

        Returns:
            Consistency scores [batch_size, seq_len, 1]
        """
        # Concatenate features
        concat = torch.cat([features1, features2], dim=-1)

        # Compute similarity
        consistency = self.similarity_net(concat)

        return consistency


class AdvancedMultiModalFusion(nn.Module):
    """
    Advanced Multi-Modal Fusion supporting Vision + Text + Audio + Video.

    This extends the basic MultiModalFusionLayer to handle additional
    modalities beyond vision and text.

    Features:
    - Support for 4 modalities: Vision, Text, Audio, Video
    - Cross-modal attention between all modality pairs
    - Confidence-based weighting
    - Temporal encoding for video
    - Cross-modal consistency validation
    - Adaptive fusion based on available modalities

    Usage:
        # With all modalities
        fusion = AdvancedMultiModalFusion()
        result = fusion(
            vision_features=vision_feats,
            text_features=text_feats,
            audio_features=audio_feats,
            video_features=video_feats,
        )

        # With subset of modalities (others can be None)
        result = fusion(
            vision_features=vision_feats,
            text_features=text_feats,
        )
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 768,
        fusion_dim: int = 768,
        num_heads: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize advanced multi-modal fusion.

        Args:
            vision_dim: Vision feature dimension
            text_dim: Text feature dimension
            audio_dim: Audio feature dimension
            video_dim: Video feature dimension
            fusion_dim: Fused feature dimension
            num_heads: Number of attention heads
            num_layers: Number of fusion layers
            dropout: Dropout rate
        """
        super().__init__()

        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        logger.info(
            f"Initializing AdvancedMultiModalFusion: "
            f"fusion_dim={fusion_dim}, num_heads={num_heads}, "
            f"num_layers={num_layers}"
        )

        # Modality encoders
        self.vision_projection = nn.Linear(vision_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.audio_encoder = AudioFeatureEncoder(audio_dim, fusion_dim)
        self.video_encoder = VideoFeatureEncoder(video_dim, fusion_dim)

        # Cross-modal attention layers
        # Vision ↔ Text
        self.vision_text_attn = CrossAttention(fusion_dim, num_heads, dropout)
        self.text_vision_attn = CrossAttention(fusion_dim, num_heads, dropout)

        # Vision ↔ Audio
        self.vision_audio_attn = CrossAttention(fusion_dim, num_heads, dropout)
        self.audio_vision_attn = CrossAttention(fusion_dim, num_heads, dropout)

        # Text ↔ Audio
        self.text_audio_attn = CrossAttention(fusion_dim, num_heads, dropout)
        self.audio_text_attn = CrossAttention(fusion_dim, num_heads, dropout)

        # Video fusion (combines with vision features)
        self.video_fusion_attn = CrossAttention(fusion_dim, num_heads, dropout)

        # Consistency checkers
        self.vision_text_consistency = CrossModalConsistencyChecker(fusion_dim)
        self.vision_audio_consistency = CrossModalConsistencyChecker(fusion_dim)
        self.text_audio_consistency = CrossModalConsistencyChecker(fusion_dim)

        # Modality gating (confidence-based weighting)
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 4),  # 4 modalities
            nn.Softmax(dim=-1),
        )

        # Output projection
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)

        # Normalization layers
        self.norm_vision = nn.LayerNorm(fusion_dim)
        self.norm_text = nn.LayerNorm(fusion_dim)
        self.norm_audio = nn.LayerNorm(fusion_dim)
        self.norm_video = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        audio_confidence: Optional[torch.Tensor] = None,
        video_frame_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Fuse multiple modalities.

        Args:
            vision_features: Vision features [batch_size, vision_len, vision_dim]
            text_features: Text features [batch_size, text_len, text_dim]
            audio_features: Audio features [batch_size, audio_len, audio_dim]
            video_features: Video features [batch_size, num_frames, video_dim]
            audio_confidence: Audio confidence scores [batch_size, audio_len]
            video_frame_indices: Video frame indices [batch_size, num_frames]

        Returns:
            Dictionary containing:
            - fused_features: Fused features
            - modality_weights: Confidence weights for each modality
            - consistency_scores: Cross-modal consistency scores
            - attention_maps: Attention weight tensors
        """
        # Project and normalize available modalities
        encoded_modalities = []
        modality_names = []

        if vision_features is not None:
            vision_enc = self.norm_vision(self.vision_projection(vision_features))
            encoded_modalities.append(vision_enc)
            modality_names.append("vision")
        else:
            vision_enc = None

        if text_features is not None:
            text_enc = self.norm_text(self.text_projection(text_features))
            encoded_modalities.append(text_enc)
            modality_names.append("text")
        else:
            text_enc = None

        if audio_features is not None:
            audio_enc = self.norm_audio(
                self.audio_encoder(audio_features, audio_confidence)
            )
            encoded_modalities.append(audio_enc)
            modality_names.append("audio")
        else:
            audio_enc = None

        if video_features is not None:
            video_enc = self.norm_video(
                self.video_encoder(video_features, video_frame_indices)
            )
            encoded_modalities.append(video_enc)
            modality_names.append("video")
        else:
            video_enc = None

        if not encoded_modalities:
            raise ValueError("At least one modality must be provided")

        logger.debug(f"Fusing modalities: {modality_names}")

        # Cross-modal attention and consistency checking
        attention_maps = {}
        consistency_scores = {}

        # Vision ↔ Text
        if vision_enc is not None and text_enc is not None:
            vision_from_text, attn_vt = self.vision_text_attn(
                vision_enc, text_enc, text_enc
            )
            text_from_vision, attn_tv = self.text_vision_attn(
                text_enc, vision_enc, vision_enc
            )
            vision_enc = vision_enc + vision_from_text
            text_enc = text_enc + text_from_vision

            attention_maps["vision_text"] = attn_vt
            attention_maps["text_vision"] = attn_tv

            consistency_scores["vision_text"] = self.vision_text_consistency(
                vision_enc, text_enc
            ).mean()

        # Vision ↔ Audio
        if vision_enc is not None and audio_enc is not None:
            vision_from_audio, attn_va = self.vision_audio_attn(
                vision_enc, audio_enc, audio_enc
            )
            audio_from_vision, attn_av = self.audio_vision_attn(
                audio_enc, vision_enc, vision_enc
            )
            vision_enc = vision_enc + vision_from_audio
            audio_enc = audio_enc + audio_from_vision

            attention_maps["vision_audio"] = attn_va
            attention_maps["audio_vision"] = attn_av

            consistency_scores["vision_audio"] = self.vision_audio_consistency(
                vision_enc, audio_enc
            ).mean()

        # Text ↔ Audio
        if text_enc is not None and audio_enc is not None:
            text_from_audio, attn_ta = self.text_audio_attn(
                text_enc, audio_enc, audio_enc
            )
            audio_from_text, attn_at = self.audio_text_attn(
                audio_enc, text_enc, text_enc
            )
            text_enc = text_enc + text_from_audio
            audio_enc = audio_enc + audio_from_text

            attention_maps["text_audio"] = attn_ta
            attention_maps["audio_text"] = attn_at

            consistency_scores["text_audio"] = self.text_audio_consistency(
                text_enc, audio_enc
            ).mean()

        # Video fusion (merge with vision if available)
        if video_enc is not None and vision_enc is not None:
            vision_from_video, attn_vv = self.video_fusion_attn(
                vision_enc, video_enc, video_enc
            )
            vision_enc = vision_enc + vision_from_video
            attention_maps["vision_video"] = attn_vv

        # Combine all modalities
        # Pool each modality to same length (mean pooling)
        pooled_modalities = []

        if vision_enc is not None:
            pooled_modalities.append(vision_enc.mean(dim=1, keepdim=True))
        if text_enc is not None:
            pooled_modalities.append(text_enc.mean(dim=1, keepdim=True))
        if audio_enc is not None:
            pooled_modalities.append(audio_enc.mean(dim=1, keepdim=True))
        if video_enc is not None:
            pooled_modalities.append(video_enc.mean(dim=1, keepdim=True))

        # Pad to 4 modalities (with zeros for missing ones)
        while len(pooled_modalities) < 4:
            pooled_modalities.append(
                torch.zeros_like(pooled_modalities[0])
            )

        # Compute modality weights
        concat_pooled = torch.cat(pooled_modalities, dim=-1)
        modality_weights = self.modality_gate(concat_pooled)  # [batch_size, 1, 4]

        # Weighted combination
        weighted_features = []
        modality_idx = 0

        for i, (modality, name) in enumerate([
            (vision_enc, "vision"),
            (text_enc, "text"),
            (audio_enc, "audio"),
            (video_enc, "video")
        ]):
            if modality is not None:
                weight = modality_weights[:, :, i:i+1]
                weighted_features.append(modality * weight)

        # Concatenate weighted features
        if weighted_features:
            fused_features = torch.cat(weighted_features, dim=1)
        else:
            fused_features = encoded_modalities[0]

        # Output projection
        fused_features = self.output_projection(fused_features)

        return {
            "fused_features": fused_features,
            "vision_features": vision_enc,
            "text_features": text_enc,
            "audio_features": audio_enc,
            "video_features": video_enc,
            "modality_weights": modality_weights,
            "consistency_scores": consistency_scores,
            "attention_maps": attention_maps,
            "modality_names": modality_names,
        }
