
# -*- coding: utf-8 -*-
"""
Проект: Классификация состояния растений по мультиспектральным снимкам
Модель: ResNet-18 (предобученная на ImageNet)
Датасет: PlantVillage (mohanty/PlantVillage)
"""

# Устанавливаем необходимые библиотеки (если еще не установлены)
# pip install torch torchvision transformers datasets scikit-learn matplotlib pillow tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# 1. Установка детерминизма для воспроизводимости
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Определяем устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# 2. Загрузка данных с Hugging Face
print("Загрузка датасета PlantVillage с Hugging Face Hub...")
# Используем предопределенные сплиты, которые учитывают группировку листьев (предотвращают утечку данных)
dataset = load_dataset("mohanty/PlantVillage", "color", trust_remote_code=True)

# Смотрим на структуру данных
print(f"Сплиты датасета: {dataset.keys()}")
print(f"Количество классов: {dataset['train'].features['labels'].num_classes}")
class_names = dataset['train'].features['labels'].names
print(f"Имена классов: {class_names[:5]}... (всего {len(class_names)})") # Покажем первые 5

# 3. Предобработка изображений (Трансформации)
# Среднее и std для ImageNet, т.к. используем предобученную модель
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Трансформации для тренировочного набора (с аугментацией)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # Приводим к размеру, ожидаемому ResNet
    transforms.RandomHorizontalFlip(p=0.5), # Случайное отражение
    transforms.RandomRotation(degrees=15),  # Случайный поворот
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Изменение цвета
    transforms.ToTensor(),                   # Преобразуем в тензор [0, 1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # Нормализация
])

# Трансформации для валидации/теста (только изменение размера и нормализация)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Функция для применения трансформаций к элементам датасета
def transform_image(batch):
    batch["pixel_values"] = [train_transforms(image) for image in batch["image"]]
    return batch

def transform_image_val(batch):
    batch["pixel_values"] = [val_transforms(image) for image in batch["image"]]
    return batch

print("Применение трансформаций к данным...")
# Применяем трансформации к каждому сплиту
dataset["train"] = dataset["train"].map(transform_image, batched=True, batch_size=64)
dataset["test"] = dataset["test"].map(transform_image_val, batched=True, batch_size=64)

# Удаляем оригинальные изображения, чтобы освободить память (они нам больше не нужны)
dataset["train"] = dataset["train"].remove_columns("image")
dataset["test"] = dataset["test"].remove_columns("image")

# Переименовываем колонку labels в labels (для совместимости с PyTorch)
# В датасете она уже называется 'labels'
dataset["train"] = dataset["train"].rename_column("labels", "labels")
dataset["test"] = dataset["test"].rename_column("labels", "labels")

# Устанавливаем формат для PyTorch
dataset["train"].set_format(type='torch', columns=['pixel_values', 'labels'])
dataset["test"].set_format(type='torch', columns=['pixel_values', 'labels'])

print("Подготовка DataLoader'ов...")
# Создаем DataLoader'ы
train_loader = DataLoader(dataset["train"], batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(dataset["test"], batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print(f"Размер тренировочной выборки: {len(dataset['train'])}")
print(f"Размер тестовой выборки: {len(dataset['test'])}")

# 4. Создание модели
num_classes = len(class_names)
print(f"Загрузка предобученной модели ResNet-18 для {num_classes} классов...")

# Загружаем предобученную модель
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# Заменяем последний fully-connected слой на новый под наше количество классов
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)

# Определяем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Добавим планировщик обучения для уменьшения lr при плато
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# 5. Функция для обучения одной эпохи
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Обучение")
    for batch in loop:
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Статистика
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1

# 6. Функция для валидации
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Валидация")
        for batch in loop:
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

# 7. Цикл обучения
num_epochs = 10
best_acc = 0.0

print("Начало обучения...")
for epoch in range(num_epochs):
    print(f"\nЭпоха {epoch+1}/{num_epochs}")
    print("-" * 50)

    train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")

    val_loss, val_acc, val_f1, _, _ = validate(model, test_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    scheduler.step(val_loss)

    # Сохраняем лучшую модель
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"-> Сохранена новая лучшая модель с точностью {best_acc:.4f}")

print("\nОбучение завершено!")

# 8. Оценка лучшей модели на тестовом наборе
print("\n" + "="*50)
print("Оценка лучшей модели на тестовом наборе")
print("="*50)

# Загружаем лучшую модель
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Получаем предсказания и метки
test_loss, test_acc, test_f1, all_preds, all_labels = validate(model, test_loader, criterion, device)

print(f"\nИтоговые метрики на тестовом наборе:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Weighted F1-Score: {test_f1:.4f}")

# 9. Отчет по метрикам и Confusion Matrix
print("\n" + "="*50)
print("Детальный отчет по классам")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# Строим матрицу ошибок
print("\nПостроение матрицы ошибок...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Матрица ошибок сохранена как 'confusion_matrix.png'")

# Функция для демонстрации предсказания на одном изображении (опционально)
def predict_image(image_path, model, transforms, class_names, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transforms(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

    return class_names[predicted_class], probabilities[predicted_class].item()

print("\nПроект успешно выполнен!")
