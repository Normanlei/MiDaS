import os
import glob
import torch
import utils
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from midas_loss import ScaleAndShiftInvariantLoss
from midas_loss import compute_scale_and_shift
from midas.model_loader import default_models, load_model
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

# NOTE:
# Leverage pretrained features and freeze the encoder, train only the decoder.
# Add contrastive loss to the decoder output to encourage the model to produce stable predictions for clean vs. augmented versions of the same image

class NYUDepthV2Dataset(Dataset):
    def __init__(self, image_dir, ground_truth_base=None, transform=None):
        self.image_paths = []
        self.augmented_image_paths = []
        self.depth_paths = []
        self.transform = transform

        # Get list of scene folders
        scene_dirs = glob.glob(os.path.join(image_dir, "*"))

        for scene_path in scene_dirs:
            scene_name = os.path.basename(scene_path)
            input_dir = os.path.join(scene_path, "input")

            # Ground truth path — mapped to original directory
            if ground_truth_base:
                gt_dir = os.path.join(ground_truth_base, scene_name, "ground_truth")
            else:
                gt_dir = os.path.join(scene_path, "ground_truth")
            
            augmented_input_dir = input_dir.replace("official_splits", "official_splits_augmented")
            input_images = sorted(glob.glob(os.path.join(input_dir, "*")))
            augmented_input_images = sorted(glob.glob(os.path.join(augmented_input_dir, "*")))
            depth_maps = sorted(glob.glob(os.path.join(gt_dir, "*.pfm")))
            # print(input_dir, augmented_input_dir, gt_dir)
            if len(input_images) != len(depth_maps) or len(augmented_input_images) != len(depth_maps) or len(input_images) != len(augmented_input_images):
                raise ValueError(f"Image/Augmented/depth count mismatch with {len(input_images)} vs {len(augmented_input_images)} vs {len(depth_maps)}")

            self.image_paths.extend(input_images)
            self.augmented_image_paths.extend(augmented_input_images)
            self.depth_paths.extend(depth_maps)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = utils.read_image(self.image_paths[idx])
        augmented_img = utils.read_image(self.augmented_image_paths[idx])
        depth, _ = utils.read_pfm(self.depth_paths[idx])
        if self.transform:
            img = self.transform({"image": img})["image"]
            augmented_img = self.transform({"image": augmented_img})["image"]
        depth = torch.from_numpy(depth.astype(np.float32))
        return img, augmented_img, depth

# --- Main ---
if __name__ == "__main__":
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    REG_CONSISTENCY_TERM = 0.05  # Regularization term for consistency loss
    TRAINING_RATIO = 0.85
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_type = 'dpt_hybrid_384'
    model_path = default_models[model_type]
    model, transform, _, _ = load_model(device, model_path, model_type, optimize=False)

    # Load original, augmented, depth_groundtruth datasets
    print("Loading original, augmented, depth_groundtruth datasets...")
    dataset = NYUDepthV2Dataset(
        image_dir='../../data/official_splits/train',
        transform=transform
    )
        
    train_size = int(TRAINING_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # # Freeze encoder (pretrained) parameters
    # print("Freezing encoder parameters...")
    # for name, param in model.pretrained.named_parameters():
    #     param.requires_grad = False
    #     # print(f"Freeze Parameter: {name}, requires_grad: {param.requires_grad}")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-6)
    loss_fn = ScaleAndShiftInvariantLoss()

    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        for img, augmented_img, depth in tqdm(train_loader):
            img, augmented_img, depth = img.to(device), augmented_img.to(device), depth.to(device)
            target_size = depth.shape[1:]

            prediction = model(img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
            ).squeeze()
            
            augmented_prediction = model(augmented_img)
            augmented_prediction = torch.nn.functional.interpolate(
                augmented_prediction.unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
            ).squeeze()

            mask = (depth > 0)
            scale1, shift1 = compute_scale_and_shift(prediction, depth, mask)
            prediction_aligned = scale1.view(-1, 1, 1) * prediction + shift1.view(-1, 1, 1)
            scale2, shift2 = compute_scale_and_shift(augmented_prediction, depth, mask)
            augmented_prediction_aligned = scale2.view(-1, 1, 1) * augmented_prediction + shift2.view(-1, 1, 1)

            
            loss_consistency = F.l1_loss(prediction_aligned, augmented_prediction_aligned)
            
            # Apply contrastive loss only after warm-up epochs:
            if epoch >= 5:
                loss = loss_fn(prediction, depth, mask) + REG_CONSISTENCY_TERM * loss_consistency
            else:
                loss = loss_fn(prediction, depth, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        scheduler.step(epoch + 1)
        print(f"📘 Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, augmented_img, depth in tqdm(val_loader):
                img, augmented_img, depth = img.to(device), augmented_img.to(device), depth.to(device)
                target_size = depth.shape[1:]

                prediction = model(img)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
                ).squeeze()
                
                augmented_prediction = model(augmented_img)
                augmented_prediction = torch.nn.functional.interpolate(
                    augmented_prediction.unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
                ).squeeze()
                
                mask = (depth > 0)
                scale1_val, shift1_val = compute_scale_and_shift(prediction, depth, mask)
                prediction_aligned = scale1_val.view(-1, 1, 1) * prediction + shift1_val.view(-1, 1, 1)
                scale2_val, shift2_val = compute_scale_and_shift(augmented_prediction, depth, mask)
                augmented_prediction_aligned = scale2_val.view(-1, 1, 1) * augmented_prediction + shift2_val.view(-1, 1, 1)
                    
                loss_consistency = F.l1_loss(prediction_aligned, augmented_prediction_aligned)

                
                loss = loss_fn(prediction, depth, mask) + REG_CONSISTENCY_TERM * loss_consistency
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"🧪 Epoch {epoch + 1} - Avg Val Loss: {avg_val_loss:.4f}")

            # --- Save Best Model ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"contrast_best_model_epoch{epoch+1}.pt")
                print(f"✅ Best model updated at Epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")
            
    # Plot losses after training each fold
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
    plt.title(f"Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"contrast_loss_curve.png")
    plt.close()
