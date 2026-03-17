import cv2
import numpy as np

# Читаем рентгеновский снимок (чёрно-белый)
image = cv2.imread('battery_xray.png', cv2.IMREAD_GRAYSCALE)

# Увеличиваем контраст (выделяем детали)
equalized = cv2.equalizeHist(image)

# Применяем пороговую обработку (пытаемся найти яркие области)
_, binary = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)

# Ищем контуры (может, это мембрана?)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Рисуем найденные контуры на цветной копии
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)

# Показываем результат
cv2.imshow('Contours', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохраняем результат
cv2.imwrite('battery_with_contours.png', image_color)