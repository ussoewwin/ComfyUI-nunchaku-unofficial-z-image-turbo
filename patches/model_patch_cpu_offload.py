"""Model Patch CPU-offload apply-time support.

Port of ``ComfyUI-NunchakuFluxLoraStacker``'s ``_model_patch_cpu_offload_apply``.

When a MODEL_PATCH is loaded with ``cpu_offload=True`` (load_device=cpu), the
stock ``comfy_extras.nodes_model_patch.ZImageControlPatch`` would feed GPU
conditioning tensors into a CPU-resident control model and crash with a device
mismatch. This patch wraps ``ZImageControlPatch.to`` / ``ZImageControlPatch.__call__``
so that, when the patch's ``load_device`` is CPU, the control computation runs
on CPU and only the residual is moved back to the GPU.

Scope matches the source: only the Z-Image Fun ControlNet apply path is
covered. The Qwen block-wise / SigLIP projector apply nodes are not patched.
"""
from __future__ import annotations

import logging

import torch

import comfy.model_management
import comfy.utils

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_hswq_cpu_offload_patched"


def apply_model_patch_cpu_offload() -> bool:
    """Install the CPU-offload apply patch. Returns True on success."""
    try:
        import comfy_extras.nodes_model_patch as _np
    except ImportError:
        logger.warning("[HSWQ ModelPatch] nodes_model_patch not found; CPU-offload apply patch skipped")
        return False

    _ZImageControlPatch = getattr(_np, "ZImageControlPatch", None)
    if _ZImageControlPatch is None:
        logger.warning("[HSWQ ModelPatch] ZImageControlPatch not found; CPU-offload apply patch skipped")
        return False

    if getattr(_ZImageControlPatch, _PATCH_FLAG, False):
        return True

    _orig_to = _ZImageControlPatch.to
    _orig_call = _ZImageControlPatch.__call__

    def _to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device) and device_or_dtype.type == "cuda":
            if getattr(self.model_patch, "load_device", None) and comfy.model_management.is_device_cpu(self.model_patch.load_device):
                return self
        return _orig_to(self, device_or_dtype)

    def _call(self, kwargs):
        if not (getattr(self.model_patch, "load_device", None) and comfy.model_management.is_device_cpu(self.model_patch.load_device)):
            return _orig_call(self, kwargs)
        x = kwargs.get("x")
        img = kwargs.get("img")
        img_input = kwargs.get("img_input")
        txt = kwargs.get("txt")
        pe = kwargs.get("pe")
        vec = kwargs.get("vec")
        block_index = kwargs.get("block_index")
        block_type = kwargs.get("block_type", "")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = None
            if self.image is not None:
                image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center").movedim(1, -1)
                self.encoded_image_size = (image_scaled.shape[-3], image_scaled.shape[-2])
            inpaint_scaled = None
            if self.inpaint_image is not None:
                inpaint_scaled = comfy.utils.common_upscale(self.inpaint_image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center").movedim(1, -1)
                self.encoded_image_size = (inpaint_scaled.shape[-3], inpaint_scaled.shape[-2])
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(image_scaled, inpaint_scaled)
            comfy.model_management.load_models_gpu(loaded_models)
        cnet_blocks = self.model_patch.model.n_control_layers
        div = round(30 / cnet_blocks)
        cnet_index = (block_index // div)
        cnet_index_float = (block_index / div)
        kwargs.pop("img")
        kwargs.pop("txt")
        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs
        dev = img.device
        if self.temp_data is None or self.temp_data[0] > cnet_index:
            enc = self.encoded_image.to("cpu").to(img.dtype)
            txt_cpu = txt.to("cpu")
            pe_cpu = pe.to("cpu") if pe is not None else pe
            vec_cpu = vec.to("cpu") if vec is not None else vec
            if block_type == "noise_refiner":
                self.temp_data = (-3, (None, self.model_patch.model(txt_cpu, enc, pe_cpu, vec_cpu)))
            else:
                self.temp_data = (-1, (None, self.model_patch.model(txt_cpu, enc, pe_cpu, vec_cpu)))
        if block_type == "noise_refiner":
            next_layer = self.temp_data[0] + 1
            inp = img_input[:, :self.temp_data[1][1].shape[1]].to("cpu")
            self.temp_data = (next_layer, self.model_patch.model.forward_noise_refiner_block(block_index, self.temp_data[1][1], inp, None, pe.to("cpu") if pe is not None else pe, vec.to("cpu") if vec is not None else vec))
            if self.temp_data[1][0] is not None:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0].to(dev) * self.strength)
        else:
            while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
                next_layer = self.temp_data[0] + 1
                inp = img_input[:, :self.temp_data[1][1].shape[1]].to("cpu")
                self.temp_data = (next_layer, self.model_patch.model.forward_control_block(next_layer, self.temp_data[1][1], inp, None, pe.to("cpu") if pe is not None else pe, vec.to("cpu") if vec is not None else vec))
            if cnet_index_float == self.temp_data[0]:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0].to(dev) * self.strength)
                if cnet_blocks == self.temp_data[0] + 1:
                    self.temp_data = None
        return kwargs

    _ZImageControlPatch.to = _to
    _ZImageControlPatch.__call__ = _call
    setattr(_ZImageControlPatch, _PATCH_FLAG, True)
    logger.info("[HSWQ ModelPatch] ZImageControlPatch CPU-offload apply patch installed")
    return True


__all__ = ["apply_model_patch_cpu_offload"]
