import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Создаём свой класс данных
class SimpleBatteryDataset(Dataset):
    def __init__(self, num_samples=10):
        # Генерируем синтетические данные
        self.images = [np.random.randn(64, 64) for _ in range(num_samples)]
        self.masks = [np.random.randint(0, 2, (64, 64)) for _ in range(num_samples)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Превращаем numpy в тензоры
        image = torch.from_numpy(self.images[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).long()
        return image.unsqueeze(0), mask  # добавляем канал: (1,64,64)

# Создаём датасет
dataset = SimpleBatteryDataset(num_samples=100)

# Создаём загрузчик (будет отдавать батчи по 16)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Проверяем
for images, masks in dataloader:
    print(f'Батч изображений: {images.shape}')
    print(f'Батч масок: {masks.shape}')
    break  # только первый батч