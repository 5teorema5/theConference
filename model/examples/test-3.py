import albumentations as A
import cv2
import numpy as np

# Загружаем настоящий снимок и его маску
image = cv2.imread('battery_xray.png', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('battery_mask.png', cv2.IMREAD_GRAYSCALE)

# Определяем аугментации (как в реальном обучении)
transform = A.Compose([
    A.RandomRotate90(p=0.5),              # поворот на 90°
    A.ShiftScaleRotate(                    # сдвиг, масштаб, поворот
        shift_limit=0.1, 
        scale_limit=0.2, 
        rotate_limit=30, 
        p=0.5
    ),
    A.RandomBrightnessContrast(p=0.3),    # изменение яркости
    A.GaussNoise(var_limit=(0.01, 0.05), p=0.3),  # шум как на рентгене
    A.ElasticTransform(p=0.2)              # "резиновая" деформация
])

# Применяем 5 раз с разными случайными параметрами
for i in range(5):
    augmented = transform(image=image, mask=mask)
    aug_image = augmented['image']
    aug_mask = augmented['mask']
    
    # Сохраняем пример
    cv2.imwrite(f'aug_image_{i}.png', aug_image)
    cv2.imwrite(f'aug_mask_{i}.png', aug_mask)
    
print("Сгенерировано 5 вариантов аугментации")