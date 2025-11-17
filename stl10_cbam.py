import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import numpy as np
import csv
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Raylu(nn.Module):
    def __init__(self, initial_sigma=3.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1) * initial_sigma)

    def forward(self, input):
        positive_sigma = torch.abs(self.sigma)
        x = torch.where(input >= 0, input, input * torch.exp(-input ** 2 / positive_sigma ** 2 / 2))
        return x


class CauchyLU(nn.Module):
    def __init__(self, initial_sigma=3.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1) * initial_sigma)

    def forward(self, input):
        positive_sigma = torch.abs(self.sigma)
        return input * ((1 / math.pi) * torch.atan(input / positive_sigma) + 0.5)


class Pgelu(nn.Module):
    def __init__(self, initial_sigma=3.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1) * initial_sigma)

    def forward(self, input):
        positive_sigma = torch.abs(self.sigma)
        temp2 = input / 2 * (1 + torch.erf(input / math.sqrt(2) / positive_sigma))
        return temp2


class Pswish(nn.Module):
    def __init__(self, initial_sigma=3.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1) * initial_sigma)

    def forward(self, input):
        positive_sigma = torch.abs(self.sigma)
        return input * torch.sigmoid(input / positive_sigma)


class Exlu(nn.Module):
    def __init__(self, initial_sigma=3.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1) * initial_sigma)

    def forward(self, input):
        input = torch.clamp(input, min=-100.0, max=100.0)
        positive_sigma = torch.clamp(torch.abs(self.sigma), min=0.1, max=10.0)
        temp1 = torch.max(torch.zeros_like(input), input)
        temp2 = torch.min(torch.zeros_like(input), input * torch.exp(input / positive_sigma))
        return temp1 + temp2


class LaplaceLU(nn.Module):
    def __init__(self, initial_sigma=3.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1) * initial_sigma)

    def forward(self, input):
        input = torch.clamp(input, min=-100.0, max=100.0)
        positive_sigma = torch.clamp(torch.abs(self.sigma), min=0.1, max=10.0)
        return torch.where(
            input >= 0,
            input * (1 - 0.5 * torch.exp(-input / positive_sigma)),
            input * 0.5 * torch.exp(input / positive_sigma)
        )


ACTIVATION_REGISTRY = {
    'raylu': Raylu,
    'cauchylu': CauchyLU, 
    'pgelu': Pgelu,
    'pswish': Pswish,
    'exlu': Exlu,
    'laplacelu': LaplaceLU,
    'relu': nn.ReLU,
}


def get_activation(name, **kwargs):
    name = name.lower()
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation function: {name}. Available: {list(ACTIVATION_REGISTRY.keys())}")
    
    activation_class = ACTIVATION_REGISTRY[name]
    
    if name == 'relu':
        return activation_class(inplace=True)
    else:
        return activation_class(**kwargs)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, dropout_rate=0.1, activation_name='raylu'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.activation1 = get_activation(activation_name)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation2 = get_activation(activation_name)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.dropout1(self.activation1(self.fc1(self.avg_pool(x)))))
        max_out = self.fc2(self.dropout2(self.activation2(self.fc1(self.max_pool(x)))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7, dropout_rate=0.1, activation_name='raylu'):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio, dropout_rate, activation_name)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ConvBlockWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_cbam=True, dropout_rate=0.1, activation_name='raylu'):
        super(ConvBlockWithCBAM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation('relu')
        self.use_cbam = use_cbam
        self.dropout_rate = dropout_rate
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        
        if use_cbam:
            self.cbam = CBAM(out_channels, dropout_rate=dropout_rate, activation_name=activation_name)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        if self.dropout_rate > 0:
            out = self.dropout(out)
        
        if self.use_cbam:
            out = self.cbam(out)
            
        return out


class SimpleCBAMNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2, activation_name='raylu'):
        super(SimpleCBAMNet, self).__init__()
        
        self.conv1 = ConvBlockWithCBAM(3, 64, use_cbam=True, dropout_rate=dropout_rate, 
                                      activation_name=activation_name)
        
        self.conv2 = ConvBlockWithCBAM(64, 256, use_cbam=True, dropout_rate=dropout_rate, 
                                      activation_name=activation_name)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_dropout = nn.Dropout(dropout_rate * 2)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.pool(x)
        
        x = self.pool(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_dropout(x)
        x = self.classifier(x)
        
        return x


def simple_xavier_init_fn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def load_stl10_data(batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    ])
    
    trainset = torchvision.datasets.STL10(
        root='./data', 
        split='train',
        download=True, 
        transform=transform_train
    )
    
    testset = torchvision.datasets.STL10(
        root='./data', 
        split='test',
        download=True, 
        transform=transform_test
    )
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def train_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total


def test_epoch(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total, test_loss / len(testloader)


def exponential_moving_average(data, alpha=0.2):
    if not data:
        return []
    
    ema = [data[0]]
    for i in range(1, len(data)):
        ema_value = alpha * data[i] + (1 - alpha) * ema[-1]
        ema.append(ema_value)
    return ema


def get_sigma_params(model):
    sigma_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'sigma' in name:
            sigma_params.append(param)
        else:
            other_params.append(param)
    
    return sigma_params, other_params


def run_single_experiment(activation_name, experiment_id, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    trainloader, testloader = load_stl10_data(batch_size)
    
    model = SimpleCBAMNet(num_classes=10, dropout_rate=0.1, activation_name=activation_name).to(device)
    model.apply(simple_xavier_init_fn)
    
    sigma_params, other_params = get_sigma_params(model)
    
    criterion = nn.CrossEntropyLoss()
    
    if sigma_params:
        sigma_lr_multiplier = 10
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': learning_rate},
            {'params': sigma_params, 'lr': learning_rate * sigma_lr_multiplier}
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    test_accs = []
    test_losses = []
    
    for epoch in range(num_epochs):
        train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        test_acc, test_loss = test_epoch(model, testloader, criterion, device)
        
        test_accs.append(test_acc)
        test_losses.append(test_loss)
    
    if len(test_accs) >= 50:
        test_accs_after_50 = test_accs[49:]
        test_losses_after_50 = test_losses[49:]
        
        ema_accs_after_50 = exponential_moving_average(test_accs_after_50, alpha=0.2)
        ema_losses_after_50 = exponential_moving_average(test_losses_after_50, alpha=0.2)
        
        final_stable_epochs = min(5, len(ema_accs_after_50))
        ema_final_avg_acc = np.mean(ema_accs_after_50[-final_stable_epochs:])
        ema_final_avg_loss = np.mean(ema_losses_after_50[-final_stable_epochs:])
        ema_max_acc = np.max(ema_accs_after_50)
        
    else:
        ema_all_accs = exponential_moving_average(test_accs, alpha=0.2)
        ema_all_losses = exponential_moving_average(test_losses, alpha=0.2)
        
        final_epochs = min(5, len(ema_all_accs))
        ema_final_avg_acc = np.mean(ema_all_accs[-final_epochs:])
        ema_final_avg_loss = np.mean(ema_all_losses[-final_epochs:])
        ema_max_acc = np.max(ema_all_accs)
        final_stable_epochs = final_epochs
    
    print(f"Experiment {experiment_id}/10:")
    print(f"EMA average accuracy (last {final_stable_epochs} epochs): {ema_final_avg_acc:.4f}%")
    print(f"EMA maximum accuracy: {ema_max_acc:.4f}%")
    print(f"EMA average loss (last {final_stable_epochs} epochs): {ema_final_avg_loss:.4f}")
    
    return ema_final_avg_acc, ema_max_acc, ema_final_avg_loss


def main():
    activation_name = 'laplacelu'
    
    num_experiments = 15
    base_seed = 42
    
    print(f"Starting {num_experiments} independent experiments")
    print(f"Activation function: {activation_name}")
    print("=" * 60)
    
    results = []
    avg_accs = []
    max_accs = []
    avg_losses = []
    
    for i in range(num_experiments):
        seed = base_seed + i
        avg_acc, max_acc, avg_loss = run_single_experiment(activation_name, i+1, seed)
        
        results.append({
            'experiment_id': i+1,
            'seed': seed,
            'avg_accuracy': avg_acc,
            'max_accuracy': max_acc,
            'avg_loss': avg_loss
        })
        
        avg_accs.append(avg_acc)
        max_accs.append(max_acc)
        avg_losses.append(avg_loss)
        
        print()
    
    mean_avg_acc = np.mean(avg_accs)
    std_avg_acc = np.std(avg_accs)
    mean_max_acc = np.mean(max_accs)
    std_max_acc = np.std(max_accs)
    mean_avg_loss = np.mean(avg_losses)
    std_avg_loss = np.std(avg_losses)
    
    print("=" * 60)
    print(f"Final Statistics ({activation_name})")
    print("=" * 60)
    print(f"Average accuracy: {mean_avg_acc:.4f}% ± {std_avg_acc:.4f}%")
    print(f"Maximum accuracy: {mean_max_acc:.4f}% ± {std_max_acc:.4f}%")
    print(f"Average loss:     {mean_avg_loss:.4f} ± {std_avg_loss:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"experiment_results_{activation_name}_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['experiment_id', 'seed', 'avg_accuracy', 'max_accuracy', 'avg_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
        
        writer.writerow({})
        writer.writerow({
            'experiment_id': 'MEAN',
            'seed': '',
            'avg_accuracy': f"{mean_avg_acc:.4f}",
            'max_accuracy': f"{mean_max_acc:.4f}",
            'avg_loss': f"{mean_avg_loss:.4f}"
        })
        writer.writerow({
            'experiment_id': 'STD',
            'seed': '',
            'avg_accuracy': f"{std_avg_acc:.4f}",
            'max_accuracy': f"{std_max_acc:.4f}",
            'avg_loss': f"{std_avg_loss:.4f}"
        })
    
    txt_filename = f"experiment_results_{activation_name}_{timestamp}.txt"
    
    with open(txt_filename, 'w', encoding='utf-8') as txtfile:
        txtfile.write(f"Experiment Report - Activation: {activation_name}\n")
        txtfile.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txtfile.write("=" * 60 + "\n\n")
        
        txtfile.write("Detailed Results:\n")
        txtfile.write("-" * 60 + "\n")
        for result in results:
            txtfile.write(f"Experiment {result['experiment_id']:2d} (seed {result['seed']:2d}): ")
            txtfile.write(f"Avg Acc={result['avg_accuracy']:6.2f}%, ")
            txtfile.write(f"Max Acc={result['max_accuracy']:6.2f}%, ")
            txtfile.write(f"Avg Loss={result['avg_loss']:6.4f}\n")
        
        txtfile.write("\n" + "=" * 60 + "\n")
        txtfile.write("Statistical Summary:\n")
        txtfile.write("-" * 60 + "\n")
        txtfile.write(f"Average accuracy: {mean_avg_acc:6.2f}% ± {std_avg_acc:6.2f}%\n")
        txtfile.write(f"Maximum accuracy: {mean_max_acc:6.2f}% ± {std_max_acc:6.2f}%\n")
        txtfile.write(f"Average loss:     {mean_avg_loss:7.4f} ± {std_avg_loss:6.4f}\n")
    
    print(f"\nResults saved to:")
    print(f"  CSV file: {csv_filename}")
    print(f"  TXT file: {txt_filename}")


if __name__ == "__main__":
    main()
