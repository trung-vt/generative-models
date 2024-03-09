from scripts.demo.streamlit_helpers import do_sample

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
