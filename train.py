import os
import glob
import torch
import utils
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from midas_loss import ScaleAndShiftInvariantLoss
from midas.model_loader import default_models, load_model
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt


class NYUDepthV2Dataset(Dataset):
    def __init__(self, image_dir, ground_truth_base=None, transform=None):
        self.image_paths = []
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

            input_images = sorted(glob.glob(os.path.join(input_dir, "*")))
            depth_maps = sorted(glob.glob(os.path.join(gt_dir, "*.pfm")))
            print(input_dir, gt_dir)
            if len(input_images) != len(depth_maps):
                raise ValueError(f"Image/depth count mismatch in {scene_path}: {len(input_images)} vs {len(depth_maps)}")

            self.image_paths.extend(input_images)
            self.depth_paths.extend(depth_maps)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = utils.read_image(self.image_paths[idx])
        depth, _ = utils.read_pfm(self.depth_paths[idx])
        if self.transform:
            img = self.transform({"image": img})["image"]
        depth = torch.from_numpy(depth.astype(np.float32))
        return img, depth

# --- Main ---
if __name__ == "__main__":
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    NUM_FOLDS = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_type = 'dpt_hybrid_384'
    model_path = default_models[model_type]
    _, transform, _, _ = load_model(device, model_path, model_type, optimize=False)

    # Load original and augmented datasets
    print("Loading original and augmented datasets...")
    # Load original dataset (ground truth is inside)
    dataset_original = NYUDepthV2Dataset(
        image_dir='../../data/official_splits/train',
        transform=transform
    )

    # Load augmented dataset (depth maps live in the original folder)
    dataset_augmented = NYUDepthV2Dataset(
        image_dir='../../data/official_splits_augmented/train',
        ground_truth_base='../../data/official_splits/train',  # redirect GT
        transform=transform
    )
    # Combine datasets
    full_dataset = ConcatDataset([dataset_original, dataset_augmented])
    print(f"Total combined samples: {len(full_dataset)}")
    
    num_samples = len(full_dataset)
    indices = torch.randperm(num_samples).tolist()
    fold_size = num_samples // NUM_FOLDS

    for fold in range(NUM_FOLDS):
        print(f"\n🔁 Starting Fold {fold + 1}/{NUM_FOLDS}")

        # Split indices for val/train
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < NUM_FOLDS - 1 else num_samples
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        # Subsets
        full_dataset.transform = transform
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # Load fresh model per fold
        model, _, _, _ = load_model(device, model_path, model_type, optimize=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = ScaleAndShiftInvariantLoss()

        best_val_loss = float('inf')
        
        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            # --- Train ---
            model.train()
            train_loss = 0
            for img, depth in train_loader:
                img, depth = img.to(device), depth.to(device)
                target_size = depth.shape[1:]

                optimizer.zero_grad()
                prediction = model(img)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
                ).squeeze()

                mask = (depth > 0)
                loss = loss_fn(prediction, depth, mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"📘 Fold {fold + 1}, Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

            # --- Validate ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for img, depth in val_loader:
                    img, depth = img.to(device), depth.to(device)
                    target_size = depth.shape[1:]

                    prediction = model(img)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
                    ).squeeze()

                    mask = (depth > 0)
                    loss = loss_fn(prediction, depth, mask)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"🧪 Fold {fold + 1}, Epoch {epoch + 1} - Avg Val Loss: {avg_val_loss:.4f}")

            # --- Save Best Model ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"best_model_fold{fold+1}.pt")
                print(f"✅ Best model updated for Fold {fold+1} at Epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

            # Save per epoch checkpoint (optional)
            # torch.save(model.state_dict(), f"midas_fold{fold+1}_epoch{epoch+1}.pt")
        # Plot losses after training each fold
        plt.figure()
        plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
        plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
        plt.title(f"Loss Curve - Fold {fold + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figs/loss_curve_fold{fold+1}.png")
        plt.close()