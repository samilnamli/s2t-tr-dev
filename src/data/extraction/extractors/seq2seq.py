"""Encoder-decoder ASR extractor.

Covers ``*ForSpeechSeq2Seq`` checkpoints — Speech2Text (Fairseq) and
Whisper. Encoder hidden states come from the model's encoder; the
decoder is only invoked for transcription via ``model.generate``.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from src.data.extraction.extractors.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class Seq2SeqExtractor(BaseFeatureExtractor):
    """Frozen :class:`AutoModelForSpeechSeq2Seq` extractor."""

    def __init__(self, *args, max_new_tokens: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.spec.hf_model_id, revision=self.spec.hf_revision
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.spec.hf_model_id, revision=self.spec.hf_revision
        )
        self.model = self._freeze(model)
        self.encoder = self.model.get_encoder()
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------

    def _processor_inputs(self, batch: List[np.ndarray]) -> dict:
        out = self.processor(
            batch,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        # Speech2Text uses ``input_features`` (mel filterbank); Whisper
        # uses ``input_features`` (log-mel). Both already at the model's
        # native frame rate.
        return {k: v.to(self.device) for k, v in out.items()}

    def _encode_batch(self, batch: List[np.ndarray]) -> List[np.ndarray]:
        if not batch:
            return []
        inputs = self._processor_inputs(batch)
        input_features = inputs["input_features"].to(self.dtype)

        kwargs = {"input_features": input_features}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"]

        encoder_out = self.encoder(**kwargs)
        hidden = encoder_out.last_hidden_state  # (B, T_enc, D)
        hidden_np = hidden.detach().to(torch.float16).cpu().numpy()

        # Whisper pads to a fixed 3000-frame mel-spectrogram (30 s) and
        # the encoder always returns a fixed number of timesteps. We
        # trim per-clip using the input-features mask when available;
        # otherwise we keep the full encoder output (Whisper's pad
        # tokens are zeros and downstream pooling handles them).
        if "attention_mask" in inputs:
            valid_lens = inputs["attention_mask"].sum(dim=-1).cpu().tolist()
            return [hidden_np[i, : int(v)] for i, v in enumerate(valid_lens)]
        return [hidden_np[i] for i in range(hidden_np.shape[0])]

    def _transcribe_batch(self, batch: List[np.ndarray]) -> List[str]:
        if not batch:
            return []
        inputs = self._processor_inputs(batch)
        input_features = inputs["input_features"].to(self.dtype)

        gen_kwargs: dict = {"max_new_tokens": self.max_new_tokens}
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs["attention_mask"]

        token_ids = self.model.generate(input_features, **gen_kwargs)
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)
