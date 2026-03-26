import time
from tqdm import tqdm

# Имитация обучения: 100 шагов
print("Обучение модели...")
for epoch in range(5):  # 5 эпох
    # Создаём прогресс-бар на 20 батчей
    for batch in tqdm(range(20), desc=f'Эпоха {epoch+1}'):
        # Имитация работы (обучение на батче)
        time.sleep(0.1)  # ждём 0.1 секунды
        
        # Можно показывать текущий loss
        if batch % 5 == 0:
            tqdm.write(f'  шаг {batch}, loss: {1.0/(batch+1):.4f}')

print("Обучение завершено!")