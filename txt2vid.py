import os

from pytorch_lightning import seed_everything

from scripts.demo.streamlit_helpers import *

import txt2vid_backend

SAVE_PATH = "outputs/demo/vid/"

VERSION2SPECS = {
    "svd": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_image_decoder": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_xt": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd_xt.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
    "svd_xt_image_decoder": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_xt_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
}

def do_UI():
    st.title("Stable Video Diffusion")


def get_model_version():
    model_version = st.selectbox(
        "Model Version",
        [k for k in VERSION2SPECS.keys()],
        0,
    )
    return model_version

def get_mode():
    if st.checkbox("Load Model"):
        mode = "img2vid"
    else:
        mode = "skip"
    return mode

def get_video_height(version_specs_dict):
    H = st.sidebar.number_input(
        "H", value=version_specs_dict["H"], min_value=64, max_value=2048
    )
    return H

def get_video_width(version_specs_dict):
    W = st.sidebar.number_input(
        "W", value=version_specs_dict["W"], min_value=64, max_value=2048
    )
    return W

def get_total_number_of_frames(version_specs_dict):
    T = st.sidebar.number_input(
        "T", value=version_specs_dict["T"], min_value=0, max_value=128
    )
    return T

def init_state(version_specs_dict):
    state = init_st(version_specs_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    return state

def get_cond_aug():
    cond_aug = st.number_input(
                "Conditioning augmentation:", value=0.02, min_value=0.0
            )
    
    return cond_aug

def get_seed():
    seed = st.sidebar.number_input(
            "seed", value=23, min_value=0, max_value=int(1e9)
        )
    
    return seed

def saving_locally():
    # TODO: Let user choose to save locally or not
    return True

def get_decoding_t(num_frames, options):
    decoding_t = st.number_input(
            "Decode t frames at a time (set small if you are low on VRAM)",
            value=options.get("decoding_t", num_frames),
            min_value=1,
            max_value=int(1e9),
        )
    
    return decoding_t

def change_fps_if_required(value_dict):
    if st.checkbox("Overwrite fps in mp4 generator", False):
        saving_fps = st.number_input(
                f"saving video at fps:", value=value_dict["fps"], min_value=1
            )
    else:
        saving_fps = value_dict["fps"]
    return saving_fps


if __name__ == "__main__":

    model_version = get_model_version()

    version_specs_dict = VERSION2SPECS[model_version]

    mode = get_mode()

    H = get_video_height(version_specs_dict)
    W = get_video_width(version_specs_dict)

    num_frames = get_total_number_of_frames(version_specs_dict)

    C = version_specs_dict["C"]
    F = version_specs_dict["f"]

    options = version_specs_dict["options"]

    if mode != "skip":
        state = init_state(version_specs_dict)
        model = state["model"]

        ukeys = set(
            get_unique_embedder_keys_from_conditioner(state["model"].conditioner)
        )

        value_dict = init_embedder_options(
            ukeys,
            {},
        )

        value_dict["image_only_indicator"] = 0

        if mode == "img2vid":
            img = load_img_for_prediction(W, H)
            cond_aug = get_cond_aug()
            value_dict["cond_frames_without_noise"] = img
            value_dict["cond_frames"] = img + cond_aug * torch.randn_like(img)
            value_dict["cond_aug"] = cond_aug

        seed = get_seed()
        seed_everything(seed)

        if saving_locally():
            save_locally, save_path = init_save_locally(
                os.path.join(SAVE_PATH, model_version), init_value=True
            )

        options["num_frames"] = num_frames

        sampler, num_rows, num_cols = init_sampling(options=options)
        num_samples = num_rows * num_cols

        decoding_t = get_decoding_t(num_frames, options)

        saving_fps = change_fps_if_required(value_dict)

        if st.button("Sample"):
            samples = txt2vid_backend.produce_sample(H, W, num_frames, C, F, options, model, value_dict, sampler, num_samples, decoding_t)

            if save_locally:
                save_video_as_grid_and_mp4(samples, save_path, num_frames, fps=saving_fps)
