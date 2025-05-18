from midas.model_loader import default_models, load_model
import os
import glob
import torch
import utils
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NYUDepthV2Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        image_classes = glob.glob(os.path.join(image_dir, "*"))
        num_image_classes = len(image_classes)
        all_image_paths = []
        all_depth_paths = []
        for index, image_class in enumerate(image_classes):
             print("  Processing {} ({}/{})".format(image_class, index + 1, num_image_classes))
             image_names = glob.glob(os.path.join(image_class, "input", "*"))
             depth_names = glob.glob(os.path.join(image_class, "ground_truth", "*.pfm"))
             all_image_paths.extend(image_names)
             all_depth_paths.extend(depth_names)
        self.image_paths = sorted(all_image_paths)
        self.depth_paths = sorted(all_depth_paths)
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = utils.read_image(self.image_paths[idx])
        depth, _ = utils.read_pfm(self.depth_paths[idx])
        
        if self.transform:
            img = self.transform({"image": img})["image"]
        
        depth = torch.from_numpy(depth.astype(np.float32))
        # print("Image shape: ", img.shape)
        # print("Depth shape: ", depth.shape)
        return img, depth


# ---- Main ----
if __name__ == "__main__":
    # Variables
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    EPOCHS = 1
    
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    
    # load model
    model_type = 'dpt_hybrid_384'
    model, transform, _, _ = load_model(device, default_models[model_type], model_type, optimize=False)
    print('Model is loaded')
    
    dataset = NYUDepthV2Dataset(image_dir='../../data/official_splits/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (img, depth) in enumerate(dataloader):
            img = img.to(device)
            depth = depth.to(device)
            target_size = depth.cpu().detach().numpy().shape[1::]
            # print("Target size: ", target_size)
                        
            optimizer.zero_grad()
            prediction = model.forward(img)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=target_size,
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
            )
            # print("Prediction shape: ", prediction.shape)
            # print("Depth shape: ", depth.shape)
            # break
            loss = loss_fn(prediction, depth)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(dataloader)}], Loss: {total_loss / 10:.4f}")
        break
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        torch.save(model.state_dict(), f"midas_finetuned_epoch{epoch+1}.pt")