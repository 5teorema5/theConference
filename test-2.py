import numpy as np
import cv2

# Создаём синтетический "рентген" (128x128, случайный шум)
xray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

# Добавляем тёмный круг в центре (имитация мембраны)
y, x = np.ogrid[:128, :128]
center_mask = (x - 64)**2 + (y - 64)**2 <= 20**2
xray[center_mask] = 50  # делаем пиксели темнее

# Создаём маску: где пиксели < 60 — это мембрана (класс 1)
mask = np.zeros((128, 128), dtype=np.uint8)
mask[xray < 60] = 1  # класс 1

# Считаем, сколько пикселей мембраны
pixels_membrane = np.sum(mask == 1)
print(f'Мембрана занимает {pixels_membrane} пикселей')

# Сохраняем маску как изображение
cv2.imwrite('synthetic_mask.png', mask * 255)  # умножаем на 255, чтобы видеть