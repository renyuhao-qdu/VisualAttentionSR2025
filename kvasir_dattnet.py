import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import math


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(47)


class EMATracker:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.ema_value = None
        self.raw_values = []
        self.ema_values = []
        
    def update(self, new_value):
        self.raw_values.append(new_value)
        
        if self.ema_value is None:
            self.ema_value = new_value
        else:
            self.ema_value = self.alpha * new_value + (1 - self.alpha) * self.ema_value
            
        self.ema_values.append(self.ema_value)
        return self.ema_value
    
    def get_current_ema(self):
        return self.ema_value
    
    def get_last_n_average(self, n=5):
        if len(self.ema_values) < n:
            return np.mean(self.ema_values) if self.ema_values else 0
        return np.mean(self.ema_values[-n:])
    
    def get_history(self):
        return self.raw_values, self.ema_values
    
    def save_to_file(self, filename_prefix):
        raw_file = f"{filename_prefix}_raw.txt"
        ema_file = f"{filename_prefix}_ema.txt"
        
        with open(raw_file, 'w') as f:
            f.write("# Raw Values\n")
            for i, val in enumerate(self.raw_values):
                f.write(f"Epoch {i+1}: {val:.6f}\n")
        
        with open(ema_file, 'w') as f:
            f.write("# EMA Values (alpha=0.5)\n")
            for i, val in enumerate(self.ema_values):
                f.write(f"Epoch {i+1}: {val:.6f}\n")
                
        print(f"Data saved to: {raw_file}, {ema_file}")


class SigmaTracker:
    def __init__(self):
        self.sigma_history = {}
        self.epochs = []
        
    def record_epoch(self, model, epoch):
        self.epochs.append(epoch)
        
        for name, module in model.named_modules():
            if hasattr(module, 'sigma'):
                if name not in self.sigma_history:
                    self.sigma_history[name] = []
                self.sigma_history[name].append(module.sigma.item())
    
    def plot_sigma_evolution(self, save_path='sigma_evolution.png'):
        if not self.sigma_history:
            print("No sigma parameter data to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, (module_name, sigma_values) in enumerate(self.sigma_history.items()):
            color = colors[i % len(colors)]
            display_name = module_name.split('.')[-1] if '.' in module_name else module_name
            plt.plot(self.epochs, sigma_values, color=color, linewidth=2, 
                    label=f'{display_name} (σ)', marker='o', markersize=3)
        
        plt.xlabel('Epoch')
        plt.ylabel('Sigma Value')
        plt.title('Sigma Parameters Evolution During Training')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nFinal Sigma Parameter Values:")
        for module_name, sigma_values in self.sigma_history.items():
            final_sigma = sigma_values[-1]
            initial_sigma = sigma_values[0]
            change = final_sigma - initial_sigma
            print(f"  {module_name}: {initial_sigma:.4f} → {final_sigma:.4f} (change: {change:+.4f})")
    
    def save_to_csv(self, filename='sigma_history.csv'):
        import pandas as pd
        
        data = {'epoch': self.epochs}
        for module_name, sigma_values in self.sigma_history.items():
            data[module_name] = sigma_values
            
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Sigma history saved to: {filename}")


USE_DATA_AUGMENTATION = False

def get_transforms(use_augmentation=False):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_target_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_target_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_target_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    return train_transform, train_target_transform, val_transform, val_target_transform


class NoiseBoostReLU(nn.Module):
    def __init__(self):
        super(NoiseBoostReLU, self).__init__()
        
    def forward(self, x):
        return F.relu(x, inplace=False)

class Raylu(nn.Module):
    def __init__(self, initial_sigma=5.0):
        super(Raylu, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma]))
    
    def forward(self, x):
        sigma_positive = torch.abs(self.sigma)
        return torch.where(x >= 0, x, x * torch.exp(-x**2 / (2 * sigma_positive**2)))

class CauchyLU(nn.Module):
    def __init__(self, initial_sigma=5.0):
        super(CauchyLU, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma]))
    
    def forward(self, x):
        sigma_positive = torch.abs(self.sigma)
        return x * ((1 / math.pi) * torch.atan(x / sigma_positive) + 0.5)

class Pgelu(nn.Module):
    def __init__(self, initial_sigma=5.0):
        super(Pgelu, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma]))
    
    def forward(self, x):
        sigma_positive = torch.abs(self.sigma)
        return 0.5 * x * (1 + torch.erf(x / (sigma_positive * math.sqrt(2))))

class Pswish(nn.Module):
    def __init__(self, initial_sigma=5.0):
        super(Pswish, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma]))
    
    def forward(self, x):
        sigma_positive = torch.abs(self.sigma)
        return x * torch.sigmoid(x / sigma_positive)

class Exlu(nn.Module):
    def __init__(self, initial_sigma=5.0):
        super(Exlu, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma]))
    
    def forward(self, x):
        x = torch.clamp(x, min=-50.0, max=50.0)
        sigma_positive = torch.clamp(torch.abs(self.sigma), min=0.1, max=10.0)
        
        pos_part = torch.max(torch.zeros_like(x), x)
        neg_part = torch.min(torch.zeros_like(x), x * torch.exp(x / sigma_positive))
        
        return pos_part + neg_part

class LaplaceLU(nn.Module):
    def __init__(self, initial_sigma=5.0):
        super(LaplaceLU, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma]))
    
    def forward(self, x):
        x = torch.clamp(x, min=-50.0, max=50.0)
        sigma_positive = torch.clamp(torch.abs(self.sigma), min=0.1, max=10.0)
        
        pos_part = x * (1 - 0.5 * torch.exp(-x / sigma_positive))
        neg_part = x * 0.5 * torch.exp(x / sigma_positive)
        
        return torch.where(x >= 0, pos_part, neg_part)


ACTIVATION_REGISTRY = {
    'relu': NoiseBoostReLU,
    'raylu': Raylu,
    'cauchylu': CauchyLU,
    'pgelu': Pgelu,
    'pswish': Pswish,
    'exlu': Exlu,
    'laplacelu': LaplaceLU,
}

GLOBAL_ACTIVATION_CONFIG = {
    'vgg_backbone': 'relu',
    'channel_attention': 'cauchylu',
    'spatial_attention': 'relu',
    'feature_fusion': 'relu',
    'context_bridge': 'relu',
    'decoder': 'relu',
}


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        reduced_channels = max(in_channels // reduction_ratio, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['channel_attention']](),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.activation = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['spatial_attention']]()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.conv(combined)
        attention = self.bn(attention)
        attention = self.activation(attention)
        attention = self.sigmoid(attention)
        
        return x * attention

class DualAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(DualAttentionModule, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(in_channels)
        self.fusion_activation = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['feature_fusion']]()
        
    def forward(self, x):
        residual = x
        
        channel_attended = self.channel_attention(x)
        spatial_attended = self.spatial_attention(channel_attended)
        
        fused_features = self.fusion_conv(spatial_attended)
        fused_features = self.fusion_bn(fused_features)
        fused_features = self.fusion_activation(fused_features)
        
        return residual + fused_features

class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.act1_1 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.act1_2 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dam1 = DualAttentionModule(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.act2_1 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.act2_2 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dam2 = DualAttentionModule(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.act3_1 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.act3_2 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.act3_3 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dam3 = DualAttentionModule(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.act4_1 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.act4_2 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.act4_3 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dam4 = DualAttentionModule(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.act5_1 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.act5_2 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.act5_3 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['vgg_backbone']]()
        self.pool5 = nn.MaxPool2d(2, 2)
        self.dam5 = DualAttentionModule(512)
        
    def forward(self, x):
        x = self.act1_1(self.bn1_1(self.conv1_1(x)))
        x = self.act1_2(self.bn1_2(self.conv1_2(x)))
        x1 = self.dam1(x)
        x = self.pool1(x1)
        
        x = self.act2_1(self.bn2_1(self.conv2_1(x)))
        x = self.act2_2(self.bn2_2(self.conv2_2(x)))
        x2 = self.dam2(x)
        x = self.pool2(x2)
        
        x = self.act3_1(self.bn3_1(self.conv3_1(x)))
        x = self.act3_2(self.bn3_2(self.conv3_2(x)))
        x = self.act3_3(self.bn3_3(self.conv3_3(x)))
        x3 = self.dam3(x)
        x = self.pool3(x3)
        
        x = self.act4_1(self.bn4_1(self.conv4_1(x)))
        x = self.act4_2(self.bn4_2(self.conv4_2(x)))
        x = self.act4_3(self.bn4_3(self.conv4_3(x)))
        x4 = self.dam4(x)
        x = self.pool4(x4)
        
        x = self.act5_1(self.bn5_1(self.conv5_1(x)))
        x = self.act5_2(self.bn5_2(self.conv5_2(x)))
        x = self.act5_3(self.bn5_3(self.conv5_3(x)))
        x5 = self.dam5(x)
        
        return x5, [x1, x2, x3, x4, x5]

class ContextFusionBridge(nn.Module):
    def __init__(self, feature_channels_list, fusion_channels=512):
        super(ContextFusionBridge, self).__init__()
        
        self.transform_layers = nn.ModuleList()
        for channels in feature_channels_list:
            self.transform_layers.append(nn.Sequential(
                nn.Conv2d(channels, fusion_channels, 1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['context_bridge']]()
            ))
        
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(fusion_channels * len(feature_channels_list), fusion_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['context_bridge']](),
            nn.Conv2d(fusion_channels, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['context_bridge']]()
        )
        
    def forward(self, feature_list):
        target_size = max([f.shape[2:] for f in feature_list])
        
        transformed_features = []
        for i, feature in enumerate(feature_list):
            transformed = self.transform_layers[i](feature)
            if transformed.shape[2:] != target_size:
                transformed = F.interpolate(transformed, size=target_size, mode='bilinear', align_corners=False)
            transformed_features.append(transformed)
        
        concatenated = torch.cat(transformed_features, dim=1)
        fused_features = self.fusion_layer(concatenated)
        
        return fused_features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['decoder']]()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = ACTIVATION_REGISTRY[GLOBAL_ACTIVATION_CONFIG['decoder']]()
        
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([skip, x], dim=1)
        
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        
        x = self.spatial_attention(x)
        
        return x

class DATTNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(DATTNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.encoder = VGG16Encoder()
        
        self.context_bridge = ContextFusionBridge(
            feature_channels_list=[64, 128, 256, 512, 512],
            fusion_channels=512
        )
        
        self.decoder4 = DecoderBlock(512, 512, 512)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        bottleneck, skip_features = self.encoder(x)
        
        context_fused = self.context_bridge(skip_features)
        
        d4 = self.decoder4(context_fused, skip_features[3])
        d3 = self.decoder3(d4, skip_features[2])
        d2 = self.decoder2(d3, skip_features[1])
        d1 = self.decoder1(d2, skip_features[0])
        
        output = self.final_conv(d1)
        
        return output


class KvasirDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            
        return image, mask


def load_kvasir_data(data_root="/root/autodl-tmp/Kvasir-SEG", batch_size=16, use_augmentation=False):
    images_dir = os.path.join(data_root, "images")
    masks_dir = os.path.join(data_root, "masks")
    
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.jpg")))
    
    if not mask_paths:
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    train_transform, train_target_transform, val_transform, val_target_transform = get_transforms(use_augmentation)
    
    train_dataset = KvasirDataset(train_imgs, train_masks, train_transform, train_target_transform)
    val_dataset = KvasirDataset(val_imgs, val_masks, val_transform, val_target_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def dice_coefficient(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.item()

def iou_score(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.item()


def train_dattnet(num_epochs=80, batch_size=16, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = load_kvasir_data(
        data_root="/root/autodl-tmp/Kvasir-SEG",
        batch_size=batch_size,
        use_augmentation=USE_DATA_AUGMENTATION
    )
    
    model = DATTNet(n_channels=3, n_classes=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    sigma_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'sigma' in name:
            sigma_params.append(param)
        else:
            other_params.append(param)
    
    if sigma_params:
        print(f"Found {len(sigma_params)} sigma parameters")
        optimizer = optim.Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': sigma_params, 'lr': learning_rate * 50}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss_ema = EMATracker(alpha=0.5)
    val_loss_ema = EMATracker(alpha=0.5)
    val_iou_ema = EMATracker(alpha=0.5)
    
    sigma_tracker = SigmaTracker()
    
    train_losses = []
    val_losses = []
    val_ious = []
    
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_loss_ema_val = train_loss_ema.update(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                for i in range(outputs.size(0)):
                    iou = iou_score(outputs[i:i+1], masks[i:i+1])
                    val_iou_sum += iou
        
        val_loss = val_loss / len(val_loader)
        val_iou = val_iou_sum / len(val_loader.dataset)
        
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        val_loss_ema_val = val_loss_ema.update(val_loss)
        val_iou_ema_val = val_iou_ema.update(val_iou)
        
        sigma_tracker.record_epoch(model, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f} | EMA: {train_loss_ema_val:.4f}')
        print(f'  Val Loss: {val_loss:.4f} | EMA: {val_loss_ema_val:.4f}')
        print(f'  Val IoU: {val_iou:.4f} | EMA: {val_iou_ema_val:.4f}')
        
        if sigma_params:
            sigma_values = [p.item() for p in sigma_params]
            avg_sigma = np.mean(sigma_values)
            min_sigma = np.min(sigma_values)
            max_sigma = np.max(sigma_values)
            print(f'  Sigma params - Avg: {avg_sigma:.4f}, Range: [{min_sigma:.4f}, {max_sigma:.4f}]')
        
        print('-' * 50)
        
        if val_iou_ema_val > best_val_iou:
            best_val_iou = val_iou_ema_val
            torch.save(model.state_dict(), 'best_dattnet_model_128x128.pth')
            print(f'Best model saved, EMA Val IoU: {best_val_iou:.4f}')
    
    final_train_loss = train_loss_ema.get_last_n_average(5)
    final_val_loss = val_loss_ema.get_last_n_average(5)
    final_val_iou = val_iou_ema.get_last_n_average(5)
    
    print(f"\nLast 5 epochs EMA averages:")
    print(f"  Train Loss: {final_train_loss:.4f}")
    print(f"  Val Loss: {final_val_loss:.4f}")
    print(f"  Val IoU: {final_val_iou:.4f}")
    print(f"  Best EMA IoU: {best_val_iou:.4f}")
    
    train_loss_ema.save_to_file('train_loss_128x128')
    val_loss_ema.save_to_file('val_loss_128x128')
    val_iou_ema.save_to_file('val_iou_128x128')
    
    print("\nPlotting sigma parameter evolution...")
    sigma_tracker.plot_sigma_evolution('sigma_evolution_128x128.png')
    
    sigma_tracker.save_to_csv('sigma_history_128x128.csv')
    
    _, train_losses_ema = train_loss_ema.get_history()
    _, val_losses_ema = val_loss_ema.get_history()
    _, val_ious_ema = val_iou_ema.get_history()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('DATTNet 128x128 Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_ious, label='Val IoU')
    plt.title('DATTNet 128x128 Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if sigma_tracker.sigma_history:
        epochs = sigma_tracker.epochs
        avg_sigmas = []
        for epoch in epochs:
            epoch_sigmas = []
            for module_name, sigma_values in sigma_tracker.sigma_history.items():
                if epoch < len(sigma_values):
                    epoch_sigmas.append(sigma_values[epoch])
            avg_sigmas.append(np.mean(epoch_sigmas) if epoch_sigmas else 0)
        
        plt.plot(epochs, avg_sigmas, 'b-', linewidth=2, label='Average Sigma')
        plt.title('Average Sigma Parameters Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Average Sigma Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dattnet_training_curves_with_sigma_128x128.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'\nTraining complete! Best validation IoU: {best_val_iou:.4f}')
    
    return model, best_val_iou


def predict_sample(model_path='best_dattnet_model_128x128.pth', num_samples=3):
    print("Starting prediction...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DATTNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    data_root = "/root/autodl-tmp/Kvasir-SEG"
    images_dir = os.path.join(data_root, "images")
    masks_dir = os.path.join(data_root, "masks")
    
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.jpg")))
    
    if not mask_paths:
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    
    _, _, val_transform, val_target_transform = get_transforms(False)
    
    indices = random.sample(range(len(image_paths)), min(num_samples, len(image_paths)))
    
    plt.figure(figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image = Image.open(image_paths[idx]).convert('RGB')
            mask = Image.open(mask_paths[idx]).convert('L')
            
            image_tensor = val_transform(image).unsqueeze(0).to(device)
            mask_tensor = val_target_transform(mask)
            
            output = model(image_tensor)
            prediction = torch.sigmoid(output).cpu().squeeze()
            
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(image)
            plt.title('Original Image (128x128)')
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(mask_tensor.squeeze(), cmap='gray')
            plt.title('Ground Truth (128x128)')
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(prediction, cmap='gray')
            plt.title('DATTNet Prediction (128x128)')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dattnet_predictions_128x128.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_sigma_evolution(csv_file='sigma_history_128x128.csv'):
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_file)
        print("Sigma Parameter Evolution Analysis:")
        
        module_names = [col for col in df.columns if col != 'epoch']
        
        for module_name in module_names:
            if module_name in df.columns:
                initial_val = df[module_name].iloc[0]
                final_val = df[module_name].iloc[-1]
                max_val = df[module_name].max()
                min_val = df[module_name].min()
                
                print(f"\n{module_name}:")
                print(f"  Initial value: {initial_val:.4f}")
                print(f"  Final value: {final_val:.4f}")
                print(f"  Change: {final_val - initial_val:+.4f}")
                print(f"  Change rate: {((final_val - initial_val) / initial_val * 100):+.2f}%")
                print(f"  Maximum value: {max_val:.4f}")
                print(f"  Minimum value: {min_val:.4f}")
                
    except FileNotFoundError:
        print(f"File {csv_file} not found. Please run training first.")


if __name__ == "__main__":
    print("DATTNet 128x128 with Noise-boosted Activation Functions & Sigma Tracking")
    print("=" * 80)
    print("How to modify activation functions:")
    print("1. Modify GLOBAL_ACTIVATION_CONFIG dictionary")
    print("2. Available: 'relu', 'raylu', 'cauchylu', 'pgelu', 'pswish', 'exlu', 'laplacelu'")
    print(f"3. Current: Original ReLU configuration")
    print()
    print("Training configuration:")
    print(f"1. Data augmentation: {'Enabled' if USE_DATA_AUGMENTATION else 'Disabled (deterministic)'}")
    print("2. Image size: 128×128")
    print("3. Batch size: 16")
    print("4. Epochs: 80")
    print("5. Learning rate: 1e-4 (normal params), 5e-3 (sigma params)")
    print("6. Optimizer: Adam with grouped learning rates")
    print("7. New feature: Sigma parameter tracking and visualization")
    print("=" * 80)
    
    model, best_iou = train_dattnet()
    
    print("\nStarting sample prediction...")
    predict_sample()
    
    print("\nAnalyzing sigma parameter evolution...")
    analyze_sigma_evolution()
    
    print(f"\nExperiment complete! Best IoU: {best_iou:.4f}")
    print("To switch activation functions, modify GLOBAL_ACTIVATION_CONFIG and rerun")
    print("Sigma evolution plot saved as: sigma_evolution_128x128.png")
    print("Sigma history data saved as: sigma_history_128x128.csv")
