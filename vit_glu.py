import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import math
import pandas as pd
import matplotlib.pyplot as plt
import os


class Raylu(nn.Module):
    def __init__(self, sigma=3.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, input):
        x = torch.where(input >= 0, input, input * torch.exp(-input ** 2 / self.sigma ** 2 / 2))
        return x


class CauchyLU(nn.Module):
    def __init__(self, sigma=3.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, input):
        return input * ((1 / math.pi) * torch.atan(input / self.sigma) + 0.5)


class Pgelu(nn.Module):
    def __init__(self, sigma=3.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, input):
        temp2 = input / 2 * (1 + torch.erf(input / math.sqrt(2) / self.sigma))
        return temp2


class Pswish(nn.Module):
    def __init__(self, sigma=3.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, input):
        return input * torch.sigmoid(input / self.sigma)


class Exlu(nn.Module):
    def __init__(self, sigma=3.0):
        super().__init__()
        self.sigma = max(0.1, min(10.0, sigma))
    
    def forward(self, input):
        input = torch.clamp(input, min=-100.0, max=100.0)
        temp1 = torch.max(torch.zeros_like(input), input)
        temp2 = torch.min(torch.zeros_like(input), input * torch.exp(input / self.sigma))
        return temp1 + temp2


class LaplaceLU(nn.Module):
    def __init__(self, sigma=3.0):
        super().__init__()
        self.sigma = max(0.1, min(10.0, sigma))
    
    def forward(self, input):
        input = torch.clamp(input, min=-100.0, max=100.0)
        return torch.where(
            input >= 0,
            input * (1 - 0.5 * torch.exp(-input / self.sigma)),
            input * 0.5 * torch.exp(input / self.sigma)
        )


ACTIVATION_REGISTRY = {
    'raylu': Raylu,
    'cauchylu': CauchyLU,
    'pgelu': Pgelu,
    'pswish': Pswish,
    'exlu': Exlu,
    'laplacelu': LaplaceLU,
    'relu': nn.ReLU,
    'gelu': nn.GELU,
}


class GLU_MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activation_name='pgelu', sigma=3.0, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
        if activation_name in ACTIVATION_REGISTRY:
            if activation_name in ['relu', 'gelu']:
                self.activation = ACTIVATION_REGISTRY[activation_name]()
            else:
                self.activation = ACTIVATION_REGISTRY[activation_name](sigma=sigma)
        else:
            raise ValueError(f"Unknown activation: {activation_name}")
        
        self.sigma = sigma
        self.activation_name = activation_name
    
    def forward(self, x):
        x1 = self.activation(self.fc1(x))
        x2 = self.fc2(x)
        x = x1 * x2
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, activation_name='pgelu', sigma=3.0, 
                 qkv_bias=False, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_features = int(dim * mlp_ratio * 2 / 3)
        self.mlp = GLU_MLP(dim, hidden_features, dim, activation_name, sigma, drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer_GLU(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=192, depth=4, num_heads=3, mlp_ratio=4.0,
                 activation_name='pgelu', sigma_list=None, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.activation_name = activation_name
        self.depth = depth
        
        if sigma_list is None:
            sigma_list = [3.0] * depth
        assert len(sigma_list) == depth, f"sigma_list length {len(sigma_list)} must equal depth {depth}"
        self.sigma_list = sigma_list
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                activation_name=activation_name, sigma=sigma_list[i],
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    config = {
        'img_size': 32,
        'patch_size': 4,
        'embed_dim': 192,
        'depth': 4,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'activation_name': 'pgelu',
        'sigma_list': [3.0, 3.0, 3.0, 3.0],
        'batch_size': 128,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.1,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Activation function: {config['activation_name']}")
    print(f"Depth: {config['depth']} blocks, Heads: {config['num_heads']}, Embed dim: {config['embed_dim']}")
    print(f"Sigma values per layer:")
    for i, sigma in enumerate(config['sigma_list']):
        print(f"  Layer {i+1}: σ = {sigma}")
    print("="*50)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_path = '/root/autodl-tmp'
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, 
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    model = VisionTransformer_GLU(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=10,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        activation_name=config['activation_name'],
        sigma_list=config['sigma_list'],
        drop_rate=config['drop_rate'],
        attn_drop_rate=config['attn_drop_rate']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("="*50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    for i in range(config['depth']):
        history[f'sigma_layer{i+1}'] = []
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, testloader, criterion, device)
        
        scheduler.step()
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        for i in range(config['depth']):
            history[f'sigma_layer{i+1}'].append(config['sigma_list'][i])
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print("\n" + "="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("="*50)
    
    df = pd.DataFrame(history)
    sigma_str = "_".join([f"{s:.1f}" for s in config['sigma_list']])
    csv_filename = f'vit_glu_{config["activation_name"]}_sigma[{sigma_str}].csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['epoch'], history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    sigma_title = ", ".join([f"σ{i+1}={config['sigma_list'][i]}" for i in range(config['depth'])])
    ax1.set_title(f'Loss - {config["activation_name"]} ({sigma_title})')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['epoch'], history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['epoch'], history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Accuracy - {config["activation_name"]} ({sigma_title})')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_filename = f'vit_glu_{config["activation_name"]}_sigma[{sigma_str}].png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_filename}")
    plt.close()
    
    return best_val_acc, best_val_loss


if __name__ == '__main__':
    main()
