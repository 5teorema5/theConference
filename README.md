# TheConference

### Сравнительный анализ современных архитектур нейронных сетей для интерпретации рентгеновских снимков химических источников тока

В проекте используются следующие библиотеки:

## OpenCV (cv2)
Зачем: загрузка, предобработка и сохранение изображений.

Что делаем:

- читаем рентгеновские снимки (в градациях серого)Библиотеки для сегментации рентгеновских изображений аккумуляторов

```python
import cv2
image = cv2.imread('xray.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('result.png', image)
```

## NumPy (numpy)
Зачем: работа с массивами пикселей.

Что делаем:
- преобразуем изображения в числовые массивы
- создаём и модифицируем маски (где пиксель = номер класса)
- считаем статистику (сколько пикселей мембраны)

```python
import numpy as np
mask = np.zeros((256, 256), dtype=np.uint8)
mask[image < 100] = 1  # класс "мембрана"
```

## Albumentations (albumentations)
Зачем: аугментация данных (искусственное увеличение датасета).

Что делаем:
- поворачиваем и наклоняем снимки (имитируем разное положение батареи)
- добавляем шум (имитируем качество рентгена)
- меняем яркость и контраст
- все изменения одновременно применяются и к изображению, и к маске

```python
import albumentations as A
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.GaussNoise(p=0.3)
])
augmented = transform(image=image, mask=mask)
```

## PyTorch (torch)
Зачем: основной фреймворк для создания и обучения нейросети.

Что делаем:
- храним данные в тензорах (специальный формат для GPU)
- переносим вычисления на видеокарту (если есть)
- создаём архитектуру U-Net

```python
import torch
tensor = torch.from_numpy(image).float()
if torch.cuda.is_available():
    tensor = tensor.cuda()
```

## torch.nn (nn)
Зачем: готовые слои для построения нейросети.

Что делаем:
- создаём свёрточные слои (они ищут границы мембраны)
- добавляем функции активации (ReLU)
- собираем U-Net из стандартных блоков

```python
import torch.nn as nn
conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
relu = nn.ReLU()
```

## torch.utils.data (Dataset, DataLoader)
Зачем: организация загрузки данных.

Что делаем:
- Dataset — описываем, как читать пары (изображение, маска)
- DataLoader — подаём данные в сеть батчами, перемешиваем, грузим параллельно

```python
from torch.utils.data import Dataset, DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

## tqdm
Зачем: прогресс-бар обучения.

Что делаем:
- видим, сколько эпох прошло
- видим скорость обучения (итераций в секунду)
- понимаем, сколько осталось ждать

```python
from tqdm import tqdm
for epoch in tqdm(range(num_epochs)):
    for batch in tqdm(dataloader):
        # обучение
```

## Установка всех библиотек одной командой
```bash
pip install opencv-python numpy albumentations torch torchvision tqdm
```

## Краткая схема работы
```text
[Рентген-снимки] → OpenCV → NumPy → Albumentations → PyTorch (Dataset) 
                                                           ↓
[Маски (разметка)] → OpenCV → NumPy -----------------→ PyTorch (Dataset)
                                                           ↓
                                                   DataLoader → U-Net → Обучение
                                                           ↓
                                                        tqdm (прогресс)
```
