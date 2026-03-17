import torch
import torch.nn as nn

# Создаём один свёрточный слой (будет искать края мембраны)
conv_layer = nn.Conv2d(
    in_channels=1,      # вход: 1 канал (ч/б)
    out_channels=4,     # выход: 4 фильтра (ищем 4 разных признака)
    kernel_size=3,      # размер окна 3x3 пикселя
    padding=1           # добавляем рамку, чтоб размер не менялся
)

# Создаём синтетический батч из 2 картинок 64x64
batch = torch.randn(2, 1, 64, 64)  # (2 картинки, 1 канал, 64px, 64px)

# Пропускаем через слой
output = conv_layer(batch)  # размер: (2, 4, 64, 64)
print(f'Вход: {batch.shape}')
print(f'Выход после свёртки: {output.shape}')

# Добавляем активацию ReLU (отсекает отрицательные значения)
relu = nn.ReLU()
activated = relu(output)
print(f'После ReLU: {activated.shape}')