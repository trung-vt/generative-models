# from scripts.demo.streamlit_helpers import do_sample

import math
import torch
from torch import autocast
import streamlit as st
from typing import List, Optional

from scripts.demo.streamlit_helpers import (
    default, get_batch, load_model, unload_model, make_grid, repeat, rearrange,
    get_unique_embedder_keys_from_conditioner
)
from sgm.modules.diffusionmodules.guiders import (LinearPredictionGuider,
                                                  VanillaCFG)

def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
    T=None,
    additional_batch_uc_fields=None,
    decoding_t=None,
):
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])

    st.text("Sampling")

    outputs = st.empty()
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]

                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                unload_model(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )
                    if k in ["crossattn", "concat"] and T is not None:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=T)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=T)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=T)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=T)

                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert T is not None

                        if isinstance(
                            sampler.guider, (VanillaCFG, LinearPredictionGuider)
                        ):
                            additional_model_inputs[k] = torch.zeros(
                                num_samples[0] * 2, num_samples[1]
                            ).to("cuda")
                        else:
                            additional_model_inputs[k] = torch.zeros(num_samples).to(
                                "cuda"
                            )
                    else:
                        additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                load_model(model.first_stage_model)
                model.en_and_decode_n_samples_a_time = (
                    decoding_t  # Decode n frames at a time
                )
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)

                if filter is not None:
                    samples = filter(samples)

                if T is None:
                    grid = torch.stack([samples])
                    grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                    outputs.image(grid.cpu().numpy())
                else:
                    as_vids = rearrange(samples, "(b t) c h w -> b t c h w", t=T)
                    for i, vid in enumerate(as_vids):
                        grid = rearrange(make_grid(vid, nrow=4), "c h w -> h w c")
                        st.image(
                            grid.cpu().numpy(),
                            f"Sample #{i} as image",
                        )

                if return_latents:
                    return samples, samples_z
                return samples



def produce_sample(H, W, num_frames, C, F, options, model, value_dict, sampler, num_samples, decoding_t):
    out = do_sample(
            model,
            sampler,
            value_dict,
            num_samples,
            H,
            W,
            C,
            F,
            T=num_frames,
            batch2model_input=["num_video_frames", "image_only_indicator"],
            force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
            force_cond_zero_embeddings=options.get(
                "force_cond_zero_embeddings", None
            ),
            return_latents=False,
            decoding_t=decoding_t,
        )

    if isinstance(out, (tuple, list)):
        samples, samples_z = out
    else:
        samples = out
        samples_z = None
    return samples
