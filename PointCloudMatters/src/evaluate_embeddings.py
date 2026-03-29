"""
Evaluation script to test PointNet++ embeddings.
Includes reconstruction quality, linear probing, and t-SNE visualization.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.models.components.pcd_encoder.pointnet2 import PointNet2
from src.models.components.pcd_encoder.decoder import PointCloudDecoder, chamfer_distance

def generate_dummy_data(batch_size, num_points, in_channels, num_classes):
    """Generate dummy synthetic point clouds for testing."""
    coord = torch.randn(batch_size * num_points, 3)
    feat = torch.randn(batch_size * num_points, in_channels)
    
    # Create simple clusters for testing classification/t-SNE
    labels = torch.randint(0, num_classes, (batch_size,))
    for i in range(batch_size):
        class_idx = labels[i].item()
        # Shift coords based on class to create distinct clusters
        coord[i*num_points:(i+1)*num_points] += class_idx * 5.0
        
    offset = torch.arange(1, batch_size + 1) * num_points
    offset = offset.to(torch.int32)
    grid_coord = (coord * 100).int()
    
    input_dict = {
        "coord": coord.cuda(),
        "feat": feat.cuda(),
        "offset": offset.cuda(),
        "grid_coord": grid_coord.cuda(),
    }
    
    return input_dict, labels.cuda()

def get_global_embedding(model, input_dict):
    """Run PointNet++ encoder and extract the global 1024-dim embedding from SA3."""
    # Temporarily hook into SA3 to get the global embedding
    global_emb = None
    
    def hook_fn(module, input, output):
        nonlocal global_emb
        _, global_emb, _ = output
        
    handle = model.sa3.register_forward_hook(hook_fn)
    _ = model(input_dict)
    handle.remove()
    
    return global_emb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path to PointNet++ weights")
    parser.add_argument("--in_channels", type=int, default=70, help="Input feature channels (e.g. 64 DINO + 6 RGB+XYZ)")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")
    
    # 1. Initialize models
    print("\n--- Initializing Models ---")
    encoder = PointNet2(in_channels=args.in_channels, num_classes=0, pretrained_path=args.ckpt).to(device)
    encoder.eval()
    
    decoder = PointCloudDecoder(embedding_dim=1024, num_points=args.num_points).to(device)
    linear_probe = nn.Linear(1024, args.num_classes).to(device)
    
    input_dict, labels = generate_dummy_data(args.batch_size, args.num_points, args.in_channels, args.num_classes)
    
    # 2. Extract Global Embeddings
    print("\n--- Extracting Embeddings ---")
    with torch.no_grad():
        global_emb = get_global_embedding(encoder, input_dict) # (B, 1024)
    print(f"Extracted global embedding shape: {global_emb.shape}")
    
    # 3. Reconstruction Evaluation (Training Decoder)
    print("\n--- Training Decoder (Reconstruction) ---")
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    
    # Reshape original coordinates to (B, N, 3) for comparison
    orig_coords = input_dict["coord"].view(args.batch_size, args.num_points, 3)
    
    decoder.train()
    for step in range(50):
        optimizer.zero_grad()
        reconstructed = decoder(global_emb) # (B, N, 3)
        
        loss = chamfer_distance(orig_coords, reconstructed).mean()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step:02d} | Chamfer Distance: {loss.item():.4f}")
            
    print(f"Final Chamfer Distance: {loss.item():.4f}")
    
    # 4. Linear Probing
    print("\n--- Training Linear Probe (Classification) ---")
    probe_opt = optim.Adam(linear_probe.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    linear_probe.train()
    for step in range(50):
        probe_opt.zero_grad()
        logits = linear_probe(global_emb.detach()) # global_emb detached because probe only
        loss = criterion(logits, labels)
        loss.backward()
        probe_opt.step()
        
        if step % 10 == 0:
            acc = (logits.argmax(dim=1) == labels).float().mean()
            print(f"Step {step:02d} | Probe Loss: {loss.item():.4f} | Accuracy: {acc.item():.2%}")
            
    # 5. t-SNE Visualization
    print("\n--- Generating t-SNE Visualization ---")
    emb_np = global_emb.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Need enough samples for t-SNE perplexity (default 30)
    if args.batch_size > 5:
        tsne = TSNE(n_components=2, perplexity=min(30, args.batch_size - 1), random_state=42)
        emb_2d = tsne.fit_transform(emb_np)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_np, cmap='tab10', alpha=0.8)
        plt.colorbar(scatter, label="Class")
        plt.title("t-SNE of PointNet++ Embeddings")
        
        os.makedirs("outputs", exist_ok=True)
        save_path = "outputs/tsne_embeddings.png"
        plt.savefig(save_path)
        print(f"Saved t-SNE plot to {save_path}")
    else:
        print("Batch size too small for meaningful t-SNE, skipping plot.")
        
    print("\nEvaluation completed successfully! ✅")

if __name__ == "__main__":
    main()
