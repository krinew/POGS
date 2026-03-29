from typing import Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import moviepy as mpy
import wandb
from pogs.tracking.observation import PosedObservation, Frame
from torchvision.transforms.functional import to_pil_image
from pogs.pogs import POGSModel
import time
import cv2


def generate_videos(frames_dict, fps=30, config_path=None):
    import datetime
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for key in frames_dict.keys():
        frames = frames_dict[key]
        if len(frames)>1:
            if frames[0].max() > 1:
                frames = [f for f in frames]
            else:
                frames = [f*255 for f in frames]
        clip = mpy.ImageSequenceClip(frames, fps=fps)
        if config_path is None:
            clip.write_videofile(f"{timestr}/{key}.mp4", codec="libx264")
        else:
            path = config_path.joinpath(f"{timestr}")
            if not path.exists():
                path.mkdir(parents=True)
            clip.write_videofile(str(path.joinpath(f"{key}.mp4")), codec="libx264")
        try:
            wandb.log({f"{key}": wandb.Video(str(path.joinpath(f"{key}.mp4")))})
        except:
            pass
    return timestr
    

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined