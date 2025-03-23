import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return x

# Training loop with warmup


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters (keep previous improved parameters)
    image_size = 32
    patch_size = 4
    in_channels = 3
    num_classes = 10
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    batch_size = 128
    learning_rate = 3e-4
    num_epochs = 50
    weight_decay = 0.05
    warmup_epochs = 5

    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Load dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                    transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                   transform=test_transform)

    # Set num_workers to 0 if issues persist
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    # Enhanced ViT model
    class ViT(nn.Module):
        def __init__(self, image_size=32, patch_size=4, in_channels=3,
                    embed_dim=512, num_heads=8, num_layers=6, num_classes=10):
            super().__init__()
            self.patch_size = patch_size
            num_patches = (image_size // patch_size) ** 2
            
            # Patch embedding
            self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                        kernel_size=patch_size, stride=patch_size)
            
            # Class token and positional encoding
            self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_encoder = PositionalEncoding(embed_dim, num_patches + 1)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim,
                dropout=0.1, activation='gelu', batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # LayerNorm and classifier
            self.norm = nn.LayerNorm(embed_dim)
            self.classifier = nn.Linear(embed_dim, num_classes)

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            nn.init.trunc_normal_(self.class_token, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            B, C, H, W = x.shape
            x = self.patch_embed(x)  # [B, E, H/P, W/P]
            x = x.flatten(2).transpose(1, 2)  # [B, N, E]
            
            # Add class token
            class_token = self.class_token.expand(B, -1, -1)
            x = torch.cat([class_token, x], dim=1)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer encoder
            x = self.transformer(x)
            
            # Classifier
            x = self.norm(x[:, 0])
            x = self.classifier(x)
            return x

    # Initialize model
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Learning rate warmup
        if epoch < warmup_epochs:
            lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Update learning rate
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Validation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        # Print metrics
        train_acc = 100 * correct / total
        test_acc = 100 * test_correct / test_total
        print(f'Epoch {epoch+1}/{num_epochs}: '
            f'Train Loss: {total_loss/len(train_loader):.4f} | '
            f'Train Acc: {train_acc:.2f}% | '
            f'Test Loss: {test_loss/len(test_loader):.4f} | '
            f'Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                'accuracy': test_acc
            }, f'best_model_epoch.pth')


    print('Training completed')
