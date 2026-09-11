"""OpenPI PyTorch prefix extraction for RL Token learning.

The observation builder must apply the checkpoint's camera transforms, tokenizer and
normalization before returning a batched openpi.models.model.Observation.
"""

from __future__ import annotations

import torch


class OpenPIFrozenFeatures:
    def __init__(self, model, observation_builder, sampling_steps: int = 10) -> None:
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        self.model = model.eval().requires_grad_(False)
        self.observation_builder = observation_builder
        self.sampling_steps = sampling_steps

    @torch.no_grad()
    def __call__(self, request):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = self.model
        model.eval()
        observation = self.observation_builder(request)
        images, image_masks, language, language_mask, _ = model._preprocess_observation(
            observation,
            train=False,
        )
        embeddings, valid, attention = model.embed_prefix(
            images, image_masks, language, language_mask
        )
        mask = model._prepare_attention_masks_4d(make_att_2d_masks(valid, attention))
        language_model = model.paligemma_with_expert.paligemma.language_model
        dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        old_attention = language_model.config._attn_implementation
        try:
            language_model.config._attn_implementation = "eager"
            (prefix, _), _ = model.paligemma_with_expert.forward(
                attention_mask=mask,
                position_ids=valid.long().cumsum(dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[embeddings.to(dtype), None],
                use_cache=False,
            )
            reference = model.sample_actions(
                observation.state.device, observation, num_steps=self.sampling_steps
            )
        finally:
            language_model.config._attn_implementation = old_attention
        return prefix.float().detach(), valid.detach(), reference.float().detach()
