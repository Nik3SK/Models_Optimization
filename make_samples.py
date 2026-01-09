import os
import random
import shutil

# Пути к исходным и целевым папкам
src_negative = 'negative'
src_positive = 'positive'
dst_negative = 'sample_negative'
dst_positive = 'sample_positive'

# Количество изображений для выборки
sample_size = 100

def create_sample(src_folder, dst_folder, n):
    # Получаем список всех файлов в папке (игнорируем подпапки)
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    if len(files) < n:
        raise ValueError(f"В папке {src_folder} меньше {n} файлов. Найдено: {len(files)}")
    
    # Выбираем случайные файлы без повторений
    selected_files = random.sample(files, n)
    
    # Создаём целевую папку, если её нет
    os.makedirs(dst_folder, exist_ok=True)
    
    # Копируем выбранные файлы
    for file in selected_files:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dst_folder, file)
        shutil.copy2(src_path, dst_path)  # copy2 сохраняет метаданные

# Выполняем выборку
create_sample(src_negative, dst_negative, sample_size)
create_sample(src_positive, dst_positive, sample_size)

print("Выборка успешно создана!")