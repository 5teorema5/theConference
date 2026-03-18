import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm  # для красивого прогресс-бара

# --- 1. Модель (U-Net) ---
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        # Здесь будет архитектура U-Net (кодировщик + декодировщик)
        # Я опустил детали для краткости, но они есть в open source
        pass

# --- 2. Датасет с аугментацией ---
class BatteryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.images = image_paths
        self.masks = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # mask должен быть типа long (классы)
        mask = mask.long()
        return image, mask

# --- 3. Трансформации (аугментации) ---
transform = A.Compose([
    A.Resize(256, 256),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.5), std=(0.5)),
    ToTensorV2()
])

# --- 4. Загрузка данных ---
dataset = BatteryDataset(
    image_paths=['img1.png', 'img2.png', ...],
    mask_paths=['mask1.png', 'mask2.png', ...],
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# --- 5. Инициализация ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # можно заменить на DiceLoss

# --- 6. ЦИКЛ ОБУЧЕНИЯ (главная часть) ---
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # переводим модель в режим обучения
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
        images = images.to(device)
        masks = masks.to(device)

        # Шаг 1: forward pass
        outputs = model(images)  # ❗ здесь сеть считает предсказания
        loss = criterion(outputs, masks)

        # Шаг 2: backward pass
        optimizer.zero_grad()   # обнуляем старые градиенты
        loss.backward()         # ❗ вычисляем градиенты (производные)
        optimizer.step()        # ❗ обновляем веса

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}')