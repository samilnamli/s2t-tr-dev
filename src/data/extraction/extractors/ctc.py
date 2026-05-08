"""CTC encoder-only ASR extractor.

Covers the ``Wav2Vec2``-family architectures (wav2vec2, HuBERT,
data2vec-audio, WavLM) when loaded via ``AutoModelForCTC``. The
encoder hidden states are read directly off the underlying SSL
backbone — the CTC head is only used for transcription.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from src.data.extraction.extractors.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class CTCExtractor(BaseFeatureExtractor):
    """Frozen :class:`AutoModelForCTC` extractor.

    Encoder hidden states are read from the model's SSL backbone
    submodule (``wav2vec2``, ``hubert``, ``data2vec_audio``, ``wavlm``)
    so the head's logits never sit in memory. Greedy CTC decoding for
    transcription.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from transformers import AutoModelForCTC, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.spec.hf_model_id, revision=self.spec.hf_revision
        )
        model = AutoModelForCTC.from_pretrained(
            self.spec.hf_model_id, revision=self.spec.hf_revision
        )
        self.model = self._freeze(model)
        self.backbone = _resolve_ssl_backbone(self.model)

    # ------------------------------------------------------------------

    def _processor_inputs(self, batch: List[np.ndarray]) -> dict:
        """Run the HF processor on a list of waveforms; return tensors on device.

        Wav2Vec2-family processors return ``input_values`` (B, T) plus
        an optional ``attention_mask``. We keep the mask whenever the
        processor produces it — some checkpoints (notably wav2vec2-base)
        ignore the mask, but feeding it costs nothing and yields correct
        encoder-output masking when the model honours it.
        """
        out = self.processor(
            batch,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.device) for k, v in out.items()}

    def _encode_batch(self, batch: List[np.ndarray]) -> List[np.ndarray]:
        if not batch:
            return []
        inputs = self._processor_inputs(batch)
        input_values = inputs["input_values"].to(self.dtype)
        attention_mask = inputs.get("attention_mask")

        kwargs = {"input_values": input_values}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        backbone_out = self.backbone(**kwargs)
        hidden = backbone_out.last_hidden_state  # (B, T_enc, D)

        valid_lens = _encoder_lengths(self.backbone, kwargs.get("attention_mask"), hidden.shape[1])
        hidden_np = hidden.detach().to(torch.float16).cpu().numpy()
        return [hidden_np[i, : valid_lens[i]] for i in range(hidden_np.shape[0])]

    def _transcribe_batch(self, batch: List[np.ndarray]) -> List[str]:
        if not batch:
            return []
        inputs = self._processor_inputs(batch)
        input_values = inputs["input_values"].to(self.dtype)
        attention_mask = inputs.get("attention_mask")

        kwargs = {"input_values": input_values}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        logits = self.model(**kwargs).logits  # (B, T, V)
        pred_ids = logits.argmax(dim=-1).cpu().numpy()

        # batch_decode handles CTC blank collapsing + tokenizer detokenization.
        return self.processor.batch_decode(pred_ids)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_BACKBONE_ATTRS = ("wav2vec2", "hubert", "data2vec_audio", "wavlm", "unispeech", "sew")


def _resolve_ssl_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """Find the SSL backbone submodule of a ``*ForCTC`` model.

    Different families name their backbone differently (``wav2vec2``,
    ``hubert``, ``data2vec_audio`` …); this scans the known attribute
    names so one extractor class covers all of them.
    """
    for attr in _BACKBONE_ATTRS:
        if hasattr(model, attr):
            return getattr(model, attr)
    # Fallback: many ``*ForCTC`` modules expose ``.base_model``.
    if hasattr(model, "base_model"):
        return model.base_model
    raise AttributeError(
        f"Could not find SSL backbone on {type(model).__name__}; "
        f"tried attributes {_BACKBONE_ATTRS}."
    )


def _encoder_lengths(
    backbone: torch.nn.Module,
    attention_mask: torch.Tensor | None,
    encoder_T: int,
) -> List[int]:
    """Compute valid encoder-output lengths per clip.

    The wav2vec2-family models expose ``_get_feat_extract_output_lengths``
    which converts raw-waveform attention-mask sums to encoder-output
    timesteps. If the helper or the mask is unavailable, fall back to
    the full encoder length (i.e. assume no padding).
    """
    if attention_mask is None:
        return [encoder_T] * 1 if encoder_T else [0]

    raw_lens = attention_mask.sum(dim=-1).cpu().tolist()
    helper = getattr(backbone, "_get_feat_extract_output_lengths", None)
    if helper is None:
        # Coarse approximation: scale by the ratio observed at the batch level.
        full_T = attention_mask.shape[1]
        ratio = encoder_T / max(full_T, 1)
        return [max(1, int(round(r * ratio))) for r in raw_lens]

    enc_lens = helper(torch.tensor(raw_lens))
    if isinstance(enc_lens, torch.Tensor):
        enc_lens = enc_lens.tolist()
    return [min(encoder_T, int(x)) for x in enc_lens]
