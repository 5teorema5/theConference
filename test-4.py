import torch

# Создаём тензор 2x2 (как маленькая картинка)
tensor_cpu = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f'На CPU:\n{tensor_cpu}')

# Проверяем, есть ли GPU
if torch.cuda.is_available():
    # Переносим на GPU
    tensor_gpu = tensor_cpu.cuda()
    print(f'На GPU: {tensor_gpu.device}')
    
    # Умножаем матрицы на GPU (очень быстро)
    result = torch.mm(tensor_gpu, tensor_gpu)
    print(f'Результат умножения:\n{result}')
else:
    print('GPU нет, считаем на CPU')

# В реальной задаче: images = torch.from_numpy(numpy_array).cuda()