""" Usage example:
python dino_pca_visualization.py --image_path shelf_iron.png
"""

import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pogs.data.utils.dino_dataloader import DinoDataloader, get_img_resolution
import tyro
import rich
import torch
from pogs.data.utils.dino_extractor import ViTExtractor
from pogs.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from torchvision import transforms
from typing import Tuple

def main(
    image_path: str = "shelf_iron.png",
    dino_model_type: str = "dinov2_vitl14",
    dino_stride: int = 14,
    device: str = "cuda",
    keep_cuda: bool = True,
):
    extractor = ViTExtractor(dino_model_type, dino_stride)
    image = Image.open(image_path)
    image = np.array(image)
    if image.dtype == np.uint8:
        image = np.array(image) / 255.0
        
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    h, w = get_img_resolution(image.shape[0], image.shape[1])
    preprocess = transforms.Compose([
                        transforms.Resize((h,w),antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    preproc_image = preprocess(torch.from_numpy(image).permute(2,0,1).unsqueeze(0)).to(device)
    dino_embeds = []
    for image in tqdm(preproc_image, desc="dino", total=1, leave=False):
        with torch.no_grad():
            descriptors = extractor.model.get_intermediate_layers(image.unsqueeze(0),reshape=True)[0].squeeze().permute(1,2,0)/10
            if keep_cuda:
                dino_embeds.append(descriptors)
            else:
                dino_embeds.append(descriptors.cpu().detach())
    out = dino_embeds[0]
    
    patch_h = out.shape[0]
    patch_w = out.shape[1]
    total_features = out.squeeze(0).squeeze(0).reshape(-1, out.shape[-1])
    pca = PCA(n_components=3)
    pca.fit(total_features.cpu().numpy())
    pca_features = pca.transform(total_features.cpu().numpy())

    # visualize PCA components for finding a proper threshold
    # 3 histograms for 3 components
    plt.subplot(2, 2, 1)
    plt.hist(pca_features[:, 0])
    plt.subplot(2, 2, 2)
    plt.hist(pca_features[:, 1])
    plt.subplot(2, 2, 3)
    plt.hist(pca_features[:, 2])
    plt.savefig(f"{image_path}_pca_hist.png")
    
    plt.clf()
    
    # Visualize PCA components
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                        (pca_features[:, 0].max() - pca_features[:, 0].min())

    plt.imshow(pca_features[0 : patch_h*patch_w, 0].reshape(patch_h, patch_w), cmap='gist_rainbow')
    plt.axis('off')

    plt.savefig(f"{image_path}_pca.png", bbox_inches='tight', pad_inches = 0)
    plt.margins(0,0)
    plt.show()
    
if __name__ == "__main__":
    tyro.cli(main)