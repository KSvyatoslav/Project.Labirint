import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import random
import tempfile
import warnings
import os
import atexit
from datetime import datetime
import traceback
import threading
import logging
import time
import sys
import math
import json
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from stl import mesh

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maze_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Константы
PREDEFINED_SHAPES = ["Звезда", "Круг", "Квадрат", "Треугольник", "Овал", "Многоугольник", "Кольцо", "Ромб", "Восьмиугольник"]
DEFAULT_IMAGE_SIZE = 800
DEFAULT_WALL_WIDTH_MM = 3.0
DEFAULT_WALL_HEIGHT_MM = 15.0
DEFAULT_BASE_HEIGHT_MM = 2.0
MAX_MAZE_SIZE = 1500  # Максимальный размер лабиринта
MAX_TEMP_FILES = 20   # Максимальное количество временных файлов

# Простой генератор лабиринта
class MazeGenerator:
    def __init__(self):
        self.directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
    
    def generate_maze_in_mask(self, mask: np.ndarray, wall_width_pixels: int = 2) -> np.ndarray:
        """Генерация лабиринта внутри маски"""
        h, w = mask.shape
        
        # Ограничиваем размер для производительности
        if h > MAX_MAZE_SIZE or w > MAX_MAZE_SIZE:
            scale = MAX_MAZE_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), 
                             interpolation=cv2.INTER_AREA) > 0
            h, w = mask.shape
            logger.info(f"Маска масштабирована до {h}x{w}")
        
        # Создаем сетку для лабиринта
        cell_size = max(3, min(h, w) // 80)  # Увеличиваем ячейки для скорости
        grid_h = h // cell_size
        grid_w = w // cell_size
        
        if grid_h < 3 or grid_w < 3:
            grid_h = max(3, h // 4)
            grid_w = max(3, w // 4)
            cell_size = min(h // grid_h, w // grid_w)
        
        # Масштабируем маску
        scaled_mask = cv2.resize(mask.astype(np.uint8), (grid_w, grid_h), 
                                interpolation=cv2.INTER_AREA) > 0
        
        # Инициализируем лабиринт
        maze_grid = np.ones((grid_h, grid_w), dtype=np.uint8)
        
        # Находим доступные ячейки
        available_cells = np.argwhere(scaled_mask)
        
        if len(available_cells) == 0:
            # Создаем простой лабиринт
            maze_grid[1:-1, 1:-1] = 0
            scaled_mask[1:-1, 1:-1] = True
        
        # Выбираем случайную начальную точку
        if len(available_cells) > 0:
            start_idx = random.randint(0, len(available_cells) - 1)
            start_y, start_x = available_cells[start_idx]
        else:
            start_y, start_x = 1, 1
        
        # Алгоритм Prim для скорости
        maze_grid[start_y, start_x] = 0
        frontiers = []
        
        for dy, dx in self.directions:
            ny, nx = start_y + dy, start_x + dx
            my, mx = start_y + dy // 2, start_x + dx // 2
            if (0 <= ny < grid_h and 0 <= nx < grid_w and
                0 <= my < grid_h and 0 <= mx < grid_w and
                scaled_mask[ny, nx] and maze_grid[ny, nx] == 1):
                frontiers.append((ny, nx, my, mx))
        
        while frontiers:
            idx = random.randint(0, len(frontiers) - 1)
            y, x, my, mx = frontiers.pop(idx)
            
            if maze_grid[y, x] == 1:
                maze_grid[y, x] = 0
                maze_grid[my, mx] = 0
                
                for dy, dx in self.directions:
                    ny, nx = y + dy, x + dx
                    nmy, nmx = y + dy // 2, x + dx // 2
                    if (0 <= ny < grid_h and 0 <= nx < grid_w and
                        0 <= nmy < grid_h and 0 <= nmx < grid_w and
                        scaled_mask[ny, nx] and maze_grid[ny, nx] == 1):
                        frontiers.append((ny, nx, nmy, nmx))
        
        # Масштабируем обратно
        maze = cv2.resize(maze_grid.astype(np.float32), (w, h), 
                         interpolation=cv2.INTER_NEAREST)
        maze = (maze > 0.5).astype(np.uint8)
        
        # Применяем маску
        maze = np.where(mask, maze, 1)
        
        return maze

# Генератор масок
class MaskGenerator:
    @staticmethod
    def create_shape_mask(shape_name: str, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
        """Создание маски для выбранной формы"""
        # Ограничиваем размер
        size = min(size, MAX_MAZE_SIZE)
        
        mask = np.zeros((size, size), dtype=bool)
        center_x, center_y = size // 2, size // 2
        
        shape_lower = shape_name.lower()
        
        if any(word in shape_lower for word in ['звезда', 'star']):
            img = Image.new('L', (size, size), 0)
            draw = ImageDraw.Draw(img)
            
            points = 5
            outer_radius = size * 0.4
            inner_radius = outer_radius * 0.4
            
            star_points = []
            for i in range(points * 2):
                angle = np.pi / 2 + i * np.pi / points
                radius = inner_radius if i % 2 == 1 else outer_radius
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                star_points.append((x, y))
            
            draw.polygon(star_points, fill=255)
            mask = np.array(img) > 127
        
        elif any(word in shape_lower for word in ['круг', 'circle']):
            radius = size * 0.4
            y, x = np.ogrid[-center_y:size-center_y, -center_x:size-center_x]
            mask = x**2 + y**2 <= radius**2
        
        elif any(word in shape_lower for word in ['квадрат', 'square']):
            margin = size // 5
            mask[margin:size-margin, margin:size-margin] = True
        
        elif any(word in shape_lower for word in ['треугольник', 'triangle']):
            pts = np.array([
                [center_x, size // 4],
                [size // 4, 3 * size // 4],
                [3 * size // 4, 3 * size // 4]
            ], np.int32)
            mask_img = np.zeros((size, size), dtype=np.uint8)
            cv2.fillPoly(mask_img, [pts], 255)
            mask = mask_img > 127
        
        elif any(word in shape_lower for word in ['овал', 'oval', 'эллипс']):
            radius_x = size * 0.35
            radius_y = size * 0.25
            y, x = np.ogrid[-center_y:size-center_y, -center_x:size-center_x]
            mask = (x**2 / radius_x**2) + (y**2 / radius_y**2) <= 1
        
        elif any(word in shape_lower for word in ['многоугольник', 'polygon']):
            sides = 6
            img = Image.new('L', (size, size), 0)
            draw = ImageDraw.Draw(img)
            
            radius = size * 0.4
            points = []
            for i in range(sides):
                angle = 2 * np.pi * i / sides
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
            
            draw.polygon(points, fill=255)
            mask = np.array(img) > 127
        
        elif any(word in shape_lower for word in ['кольцо', 'ring']):
            inner_radius = size * 0.2
            outer_radius = size * 0.4
            y, x = np.ogrid[-center_y:size-center_y, -center_x:size-center_x]
            r = np.sqrt(x**2 + y**2)
            mask = (r >= inner_radius) & (r <= outer_radius)
        
        elif any(word in shape_lower for word in ['ромб', 'diamond']):
            pts = np.array([
                [center_x, size // 4],
                [size // 4, center_y],
                [center_x, 3 * size // 4],
                [3 * size // 4, center_y]
            ], np.int32)
            mask_img = np.zeros((size, size), dtype=np.uint8)
            cv2.fillPoly(mask_img, [pts], 255)
            mask = mask_img > 127
        
        elif any(word in shape_lower for word in ['восьмиугольник', 'octagon']):
            img = Image.new('L', (size, size), 0)
            draw = ImageDraw.Draw(img)
            
            radius = size * 0.4
            points = []
            for i in range(8):
                angle = 2 * np.pi * i / 8
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
            
            draw.polygon(points, fill=255)
            mask = np.array(img) > 127
        
        else:
            radius = size * 0.4
            y, x = np.ogrid[-center_y:size-center_y, -center_x:size-center_x]
            mask = x**2 + y**2 <= radius**2
        
        # Улучшаем маску
        mask = mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = mask > 127
        
        return mask

# Оптимизированный генератор STL
class OptimizedSTLGenerator:
    """Оптимизированный генератор STL с объединением стен"""
    
    @staticmethod
    def maze_to_stl_optimized(maze: np.ndarray, 
                             wall_height_mm: float = 15.0,
                             wall_width_mm: float = 3.0,
                             base_height_mm: float = 2.0,
                             scale_factor: float = 1.0) -> Optional[mesh.Mesh]:
        """Создание оптимизированной 3D модели STL из лабиринта"""
        try:
            # Масштабируем параметры
            wall_height = wall_height_mm * scale_factor
            wall_width = wall_width_mm * scale_factor
            base_height = base_height_mm * scale_factor
            
            h, w = maze.shape
            
            # Автоматическая корректировка размера для печати
            max_model_size = 300  # мм
            if w * wall_width > max_model_size:
                wall_width = max_model_size / w
                logger.info(f"Ширина стен скорректирована до {wall_width:.2f} мм для печати")
            
            # Все вершины и грани
            all_vertices = []
            all_faces = []
            
            # 1. Добавляем основание (один большой прямоугольник)
            base_vertices = [
                [0, 0, 0],
                [w * wall_width, 0, 0],
                [w * wall_width, h * wall_width, 0],
                [0, h * wall_width, 0],
                [0, 0, base_height],
                [w * wall_width, 0, base_height],
                [w * wall_width, h * wall_width, base_height],
                [0, h * wall_width, base_height]
            ]
            
            base_faces = [
                [0, 3, 1], [1, 3, 2],  # низ
                [4, 5, 7], [5, 6, 7],  # верх
                [0, 1, 4], [1, 5, 4],  # бок 1
                [1, 2, 5], [2, 6, 5],  # бок 2
                [2, 3, 6], [3, 7, 6],  # бок 3
                [3, 0, 7], [0, 4, 7]   # бок 4
            ]
            
            all_vertices.extend(base_vertices)
            all_faces.extend(base_faces)
            
            # 2. Оптимизация: объединяем смежные стены
            visited = np.zeros_like(maze, dtype=bool)
            wall_rectangles = []
            
            # Сначала объединяем горизонтально
            for y in range(h):
                x = 0
                while x < w:
                    if maze[y, x] == 1 and not visited[y, x]:
                        # Находим длину горизонтальной стены
                        length = 1
                        while x + length < w and maze[y, x + length] == 1 and not visited[y, x + length]:
                            length += 1
                        
                        # Находим высоту (сколько строк имеют такую же стену)
                        height = 1
                        can_extend = True
                        while y + height < h and can_extend:
                            for i in range(length):
                                if not (maze[y + height, x + i] == 1 and not visited[y + height, x + i]):
                                    can_extend = False
                                    break
                            if can_extend:
                                height += 1
                        
                        # Отмечаем как посещенное
                        visited[y:y+height, x:x+length] = True
                        wall_rectangles.append((x, y, length, height))
                        
                        x += length
                    else:
                        x += 1
            
            # 3. Создаем призмы для объединенных стен
            vertex_offset = len(all_vertices)
            
            for x, y, length, height in wall_rectangles:
                # Создаем одну большую призму вместо множества кубов
                x_start = x * wall_width
                y_start = y * wall_width
                x_end = (x + length) * wall_width
                y_end = (y + height) * wall_width
                
                wall_vertices = [
                    [x_start, y_start, base_height],
                    [x_end, y_start, base_height],
                    [x_end, y_end, base_height],
                    [x_start, y_end, base_height],
                    [x_start, y_start, base_height + wall_height],
                    [x_end, y_start, base_height + wall_height],
                    [x_end, y_end, base_height + wall_height],
                    [x_start, y_end, base_height + wall_height]
                ]
                
                wall_faces = [
                    [0, 3, 1], [1, 3, 2],  # низ
                    [4, 5, 7], [5, 6, 7],  # верх
                    [0, 1, 4], [1, 5, 4],  # бок 1
                    [1, 2, 5], [2, 6, 5],  # бок 2
                    [2, 3, 6], [3, 7, 6],  # бок 3
                    [3, 0, 7], [0, 4, 7]   # бок 4
                ]
                
                # Добавляем с учетом смещения
                all_vertices.extend(wall_vertices)
                for face in wall_faces:
                    all_faces.append([v + vertex_offset for v in face])
                
                vertex_offset += 8
            
            logger.info(f"Создано {len(wall_rectangles)} объединенных стен (вместо {np.sum(maze == 1)} отдельных)")
            
            if len(wall_rectangles) == 0:
                logger.warning("Нет стен для создания STL модели")
                return None
            
            # 4. Конвертируем в numpy массивы и создаем mesh
            vertices_array = np.array(all_vertices, dtype=np.float32)
            faces_array = np.array(all_faces, dtype=np.int32)
            
            # Создаем STL mesh более эффективно
            data = np.zeros(faces_array.shape[0], dtype=mesh.Mesh.dtype)
            mesh_obj = mesh.Mesh(data, remove_empty_areas=False)
            
            # Заполняем векторы напрямую
            for i, face in enumerate(faces_array):
                mesh_obj.vectors[i] = vertices_array[face]
            
            return mesh_obj
            
        except Exception as e:
            logger.error(f"Ошибка при создании STL: {e}")
            traceback.print_exc()
            return None
    
    @staticmethod
    def save_stl(stl_mesh: mesh.Mesh, filepath: str) -> bool:
        """Сохранение STL модели с проверками"""
        try:
            if stl_mesh is None:
                return False
            
            stl_mesh.save(filepath)
            
            # Проверяем размер файла
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                logger.error("Создан пустой STL файл")
                return False
            
            logger.info(f"STL сохранен: {filepath} ({file_size:,} байт)")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения STL: {e}")
            return False

# Улучшенный процессор для обработки изображений
class EnhancedImageProcessor:
    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """Предварительная обработка изображения для консистентности"""
        try:
            # Если изображение с прозрачностью (RGBA), удаляем альфа-канал
            if image.shape[2] == 4:
                # Создаем белый фон
                white_bg = np.ones_like(image[:, :, :3]) * 255
                alpha = image[:, :, 3:4] / 255.0
                image = (image[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            
            # Конвертируем в RGB если нужно
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                # Проверяем порядок каналов
                if image[0, 0, 0] > image[0, 0, 2]:  # Если BGR вместо RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Ошибка предобработки изображения: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((100, 100, 3), dtype=np.uint8)

    @staticmethod
    def create_mask_from_image(image: np.ndarray, size: int, 
                              auto_invert: bool = True,
                              use_edge_detection: bool = False,
                              threshold_method: str = "otsu") -> np.ndarray:
        """
        Создание маски из изображения с поддержкой разных форматов
        
        Параметры:
        - auto_invert: автоматически инвертировать, если белый фон
        - use_edge_detection: использовать детектирование границ для сложных изображений
        - threshold_method: метод бинаризации ("otsu", "adaptive", "triangle")
        """
        try:
            # 1. Предварительная обработка
            processed = EnhancedImageProcessor.preprocess_image(image)
            
            # 2. Масштабирование
            size = min(size, MAX_MAZE_SIZE)
            processed = cv2.resize(processed, (size, size), interpolation=cv2.INTER_AREA)
            
            # 3. Конвертация в градации серого
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            else:
                gray = processed
            
            # 4. Улучшение контраста (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 5. Размытие для уменьшения шума
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 6. Бинаризация выбранным методом
            if threshold_method == "adaptive":
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
            elif threshold_method == "triangle":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            else:  # "otsu" по умолчанию
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 7. Детектирование границ для сложных изображений (опционально)
            if use_edge_detection:
                edges = cv2.Canny(gray, 50, 150)
                # Заполняем внутренние области
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    edge_mask = np.zeros_like(binary)
                    cv2.drawContours(edge_mask, contours, -1, 255, -1)
                    # Комбинируем с бинаризацией
                    binary = cv2.bitwise_and(binary, edge_mask)
            
            # 8. Автоматическое определение и инверсия если нужно
            if auto_invert:
                # Определяем, преобладает ли белый цвет (вероятно фон)
                white_ratio = np.sum(binary > 127) / (size * size)
                if white_ratio > 0.7:  # Если больше 70% белого
                    binary = cv2.bitwise_not(binary)
                    logger.info(f"Маска автоматически инвертирована (белый фон: {white_ratio:.2%})")
            
            # 9. Морфологические операции для улучшения маски
            kernel = np.ones((3, 3), np.uint8)
            
            # Закрытие для заполнения мелких отверстий
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Открытие для удаления мелкого шума
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 10. Заполнение внутренних областей для контуров
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                filled_mask = np.zeros_like(binary)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Игнорируем очень маленькие контуры
                        cv2.drawContours(filled_mask, [contour], 0, 255, -1)
                binary = filled_mask
            
            # 11. Гауссово размытие и повторная бинаризация для сглаживания
            binary = cv2.GaussianBlur(binary, (5, 5), 0)
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
            
            # 12. Убеждаемся, что есть достаточно белой области
            white_pixels = np.sum(binary > 127)
            if white_pixels < (size * size * 0.01):  # Менее 1% белого
                logger.warning(f"Маска слишком темная, используем простую бинаризацию")
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 13. Финальные морфологические операции
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            return binary > 0
            
        except Exception as e:
            logger.error(f"Ошибка создания маски из изображения: {e}")
            traceback.print_exc()
            # Возвращаем простую маску по умолчанию
            mask = np.ones((size, size), dtype=bool)
            margin = size // 4
            mask[margin:size-margin, margin:size-margin] = False
            return mask

    @staticmethod
    def create_advanced_mask(image: np.ndarray, size: int, 
                           method: str = 'auto',
                           use_grabcut_refinement: bool = True,
                           gaussian_blur_kernel: tuple = (5, 5),
                           clahe_clip_limit: float = 2.0) -> np.ndarray:
        """
        УЛУЧШЕННОЕ создание маски для сложных изображений.
        Особенно эффективно для черных, разнотонных и низкоконтрастных изображений.
        
        Параметры:
        - method: 'auto', 'adaptive', 'edge_based', 'grabcut'
        - use_grabcut_refinement: использовать GrabCut для уточнения маски
        - gaussian_blur_kernel: размер ядра для размытия по Гауссу
        - clahe_clip_limit: параметр CLAHE для улучшения контраста
        """
        try:
            # 1. Предварительная обработка (универсальная)
            processed = EnhancedImageProcessor.preprocess_image(image)
            size = min(size, MAX_MAZE_SIZE)
            processed = cv2.resize(processed, (size, size), interpolation=cv2.INTER_AREA)
            
            # 2. Улучшение контраста для сложных изображений
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            
            # Применяем CLAHE для улучшения локального контраста
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
            gray_enhanced = clahe.apply(gray)
            
            # Гауссово размытие для уменьшения шума
            gray_blurred = cv2.GaussianBlur(gray_enhanced, gaussian_blur_kernel, 0)
            
            # 3. Автоматический выбор или применение указанного метода
            if method == 'auto':
                # Анализируем гистограмму для выбора метода
                hist = cv2.calcHist([gray_blurred], [0], None, [256], [0, 256])
                contrast = np.std(gray_blurred)  # Мера контраста
                
                if contrast < 25:  # Очень низкоконтрастное изображение
                    mask = EnhancedImageProcessor._create_edge_based_mask(gray_blurred, processed)
                elif np.argmax(hist) < 50 or np.argmax(hist) > 200:  # Очень тёмное или светлое
                    mask = EnhancedImageProcessor._create_adaptive_mask(gray_blurred)
                else:
                    # Пробуем несколько методов и выбираем лучший
                    masks = []
                    masks.append(EnhancedImageProcessor._create_adaptive_mask(gray_blurred))
                    masks.append(EnhancedImageProcessor._create_edge_based_mask(gray_blurred, processed))
                    
                    # Выбираем маску с наибольшей детализацией (но не шумом)
                    best_mask = masks[0]
                    best_score = 0
                    
                    for m in masks:
                        contours, _ = cv2.findContours(m.astype(np.uint8), 
                                                      cv2.RETR_EXTERNAL, 
                                                      cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            area = sum(cv2.contourArea(c) for c in contours)
                            perimeter = sum(cv2.arcLength(c, True) for c in contours)
                            if perimeter > 0:
                                score = area / perimeter  # Мера "компактности"
                                if score > best_score and area > size*size*0.01:
                                    best_score = score
                                    best_mask = m
                    
                    mask = best_mask
            
            elif method == 'adaptive':
                mask = EnhancedImageProcessor._create_adaptive_mask(gray_blurred)
            elif method == 'edge_based':
                mask = EnhancedImageProcessor._create_edge_based_mask(gray_blurred, processed)
            elif method == 'grabcut':
                mask = EnhancedImageProcessor._create_grabcut_mask(processed)
            else:
                mask = EnhancedImageProcessor._create_adaptive_mask(gray_blurred)
            
            # 4. Уточнение маски с помощью GrabCut (если включено)
            if use_grabcut_refinement and mask.any() and not mask.all():
                mask = EnhancedImageProcessor._refine_with_grabcut(processed, mask)
            
            # 5. Постобработка маски
            mask = mask.astype(np.uint8) * 255
            
            # Морфологические операции для очистки
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Заполнение внутренних областей
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                filled_mask = np.zeros_like(mask)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        cv2.drawContours(filled_mask, [contour], 0, 255, -1)
                mask = filled_mask
            
            return mask > 127
            
        except Exception as e:
            logger.error(f"Ошибка в create_advanced_mask: {e}")
            # Возвращаем простую маску по умолчанию
            mask = np.ones((size, size), dtype=bool)
            margin = size // 4
            mask[margin:size-margin, margin:size-margin] = False
            return mask

    @staticmethod
    def _create_adaptive_mask(gray: np.ndarray) -> np.ndarray:
        """Создание маски с адаптивной бинаризацией"""
        # Адаптивная бинаризация для изображений с переменным освещением
        binary = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Автоматическая инверсия при необходимости
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        return binary > 127

    @staticmethod
    def _create_edge_based_mask(gray: np.ndarray, color_img: np.ndarray) -> np.ndarray:
        """Создание маски на основе детекции границ"""
        # Детекция границ Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Расширение границ для соединения разрывов
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Нахождение и заполнение контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(gray, dtype=bool)
        
        # Создаем маску из самых больших контуров
        mask = np.zeros_like(gray, dtype=np.uint8)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]  # Топ-3 контура
        
        for contour in contours:
            if cv2.contourArea(contour) > gray.size * 0.001:  # Минимальная площадь
                cv2.drawContours(mask, [contour], 0, 255, -1)
        
        return mask > 127

    @staticmethod
    def _create_grabcut_mask(image: np.ndarray) -> np.ndarray:
        """Создание маски с помощью алгоритма GrabCut"""
        # Инициализация маски для GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Прямоугольник по умолчанию (центр изображения)
        h, w = image.shape[:2]
        rect = (w//4, h//4, w//2, h//2)
        
        # Временные массивы для алгоритма
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Применяем GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            
            # Преобразуем маску в бинарную
            mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype(bool)
            
            # Если маска пустая или полная, пробуем другую инициализацию
            if not mask_binary.any() or mask_binary.all():
                mask[:] = 0
                cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype(bool)
            
            return mask_binary
        except:
            return np.ones(image.shape[:2], dtype=bool)

    @staticmethod
    def _refine_with_grabcut(image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """Уточнение маски с помощью GrabCut"""
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Устанавливаем начальную маску
        mask[initial_mask] = cv2.GC_PR_FGD
        mask[~initial_mask] = cv2.GC_PR_BGD
        
        # Устанавливаем уверенные области по краям
        mask[0, :] = cv2.GC_BGD
        mask[-1, :] = cv2.GC_BGD
        mask[:, 0] = cv2.GC_BGD
        mask[:, -1] = cv2.GC_BGD
        
        # Временные массивы
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
            refined_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(bool)
            return refined_mask
        except:
            return initial_mask

# Основной процессор
class MazeProcessor:
    def __init__(self):
        self.maze_gen = MazeGenerator()
        self.mask_gen = MaskGenerator()
        self.stl_gen = OptimizedSTLGenerator()
        self.img_processor = EnhancedImageProcessor()
        self.temp_files = []
        self.temp_lock = threading.Lock()
        atexit.register(self.cleanup_temp_files)
    
    def add_temp_file(self, filepath: str):
        """Добавление временного файла с ограничением количества"""
        with self.temp_lock:
            self.temp_files.append(filepath)
            if len(self.temp_files) > MAX_TEMP_FILES:
                # Удаляем самые старые файлы
                while len(self.temp_files) > MAX_TEMP_FILES // 2:
                    old_file = self.temp_files.pop(0)
                    try:
                        if os.path.exists(old_file):
                            os.unlink(old_file)
                            logger.debug(f"Удален старый временный файл: {old_file}")
                    except Exception as e:
                        logger.warning(f"Не удалось удалить файл {old_file}: {e}")
    
    def cleanup_temp_files(self):
        """Очистка временных файлов"""
        with self.temp_lock:
            for file_path in self.temp_files:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                        logger.debug(f"Очистка: удален {file_path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить файл {file_path}: {e}")
            self.temp_files.clear()
    
    def validate_inputs(self, image_size: int, wall_width_pixels: int) -> Tuple[bool, str]:
        """Валидация входных параметров"""
        if image_size < 100 or image_size > 5000:
            return False, "Размер изображения должен быть от 100 до 5000 пикселей"
        if wall_width_pixels < 1 or wall_width_pixels > 20:
            return False, "Ширина стен должна быть от 1 до 20 пикселей"
        return True, ""
    
    def process_maze(self, shape_name: str, uploaded_image=None, use_custom=False,
                    image_size: int = DEFAULT_IMAGE_SIZE, wall_width_pixels: int = 2,
                    mask_params: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Создание лабиринта"""
        try:
            # Валидация
            is_valid, error_msg = self.validate_inputs(image_size, wall_width_pixels)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Ограничиваем размер для производительности
            image_size = min(image_size, MAX_MAZE_SIZE)
            
            # Параметры маски по умолчанию
            if mask_params is None:
                mask_params = {
                    'auto_invert': True,
                    'use_edge_detection': False,
                    'threshold_method': 'otsu',
                    'advanced_method': 'auto',
                    'use_grabcut': True,
                    'clahe_limit': 2.0
                }
            
            # Создаем маску
            if use_custom and uploaded_image is not None:
                # Используем улучшенный метод для сложных изображений
                mask = self.img_processor.create_advanced_mask(
                    uploaded_image, 
                    image_size,
                    method=mask_params.get('advanced_method', 'auto'),
                    use_grabcut_refinement=mask_params.get('use_grabcut', True),
                    clahe_clip_limit=mask_params.get('clahe_limit', 2.0)
                )
            else:
                mask = self.mask_gen.create_shape_mask(shape_name, image_size)
            
            if mask is None or np.sum(mask) == 0:
                raise ValueError("Не удалось создать маску. Попробуйте другое изображение или настройки.")
            
            # Логируем статистику маски
            mask_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
            logger.info(f"Маска создана: {mask.shape}, заполнение: {mask_ratio:.2%}")
            
            # Генерируем лабиринт
            maze = self.maze_gen.generate_maze_in_mask(mask, wall_width_pixels)
            
            # Визуализация
            result_image = self.visualize_maze(maze, mask)
            
            # Статистика
            stats = self._calculate_statistics(maze, mask)
            
            return result_image, maze, mask, stats
            
        except Exception as e:
            logger.error(f"Ошибка при обработке лабиринта: {e}")
            traceback.print_exc()
            error_image = self._create_error_image(str(e))
            return error_image, None, None, {"error": str(e)}
    
    def visualize_maze(self, maze: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Визуализация лабиринта (оптимизированная)"""
        h, w = maze.shape
        
        # Используем векторные операции вместо циклов
        result = np.full((h, w, 3), [30, 30, 60], dtype=np.uint8)
        result[mask] = [240, 240, 240]
        result[maze == 1] = [20, 20, 20]
        
        return result
    
    def _calculate_statistics(self, maze: np.ndarray, mask: np.ndarray) -> Dict:
        """Расчет статистики"""
        try:
            total_area = np.sum(mask)
            wall_area = np.sum((maze == 1) & mask)
            passage_area = np.sum((maze == 0) & mask)
            
            if total_area > 0:
                wall_percentage = (wall_area / total_area) * 100
                passage_percentage = (passage_area / total_area) * 100
            else:
                wall_percentage = passage_percentage = 0
            
            return {
                "Размер лабиринта": f"{maze.shape[1]} × {maze.shape[0]} пикселей",
                "Общая площадь": f"{total_area:,} пикселей",
                "Площадь стен": f"{wall_area:,} пикселей ({wall_percentage:.1f}%)",
                "Площадь проходов": f"{passage_area:,} пикселей ({passage_percentage:.1f}%)",
                "Отношение стен/проходов": f"{wall_area/max(passage_area, 1):.2f}"
            }
        except:
            return {"error": "Не удалось рассчитать статистику"}
    
    def _create_error_image(self, message: str) -> np.ndarray:
        """Изображение с ошибкой"""
        img = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
        img[:] = [50, 50, 80]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = self._wrap_text(message, DEFAULT_IMAGE_SIZE - 100)
        
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, 0.7, 1)[0]
            text_x = (DEFAULT_IMAGE_SIZE - text_size[0]) // 2
            text_y = DEFAULT_IMAGE_SIZE // 2 + i * 30 - len(lines) * 15
            cv2.putText(img, line, (text_x, text_y), font, 0.7, (255, 200, 200), 1)
        
        return img
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Разбивка текста"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) * 12 > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines[:4]
    
    def save_png(self, image: np.ndarray) -> Optional[str]:
        """Сохранение PNG"""
        try:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.png', 
                prefix=f'maze_{datetime.now().strftime("%H%M%S")}_'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.add_temp_file(temp_path)
            
            file_size = os.path.getsize(temp_path)
            logger.info(f"PNG сохранен: {temp_path} ({file_size:,} байт)")
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Ошибка сохранения PNG: {e}")
            return None
    
    def generate_stl(self, maze: np.ndarray, wall_height_mm: float = 15.0,
                    wall_width_mm: float = 3.0, base_height_mm: float = 2.0,
                    scale_factor: float = 1.0) -> Optional[str]:
        """Генерация STL"""
        try:
            if maze is None:
                logger.warning("Нет данных лабиринта для генерации STL")
                return None
            
            # Проверяем, есть ли стены в лабиринте
            wall_count = np.sum(maze == 1)
            if wall_count == 0:
                logger.warning("Нет стен в лабиринте для создания STL")
                return None
            
            logger.info(f"Начинаем генерацию STL для лабиринта с {wall_count} стен...")
            start_time = time.time()
            
            stl_mesh = self.stl_gen.maze_to_stl_optimized(
                maze, wall_height_mm, wall_width_mm, base_height_mm, scale_factor)
            
            if stl_mesh is None:
                return None
            
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.stl', 
                prefix=f'maze_3d_{datetime.now().strftime("%H%M%S")}_'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            if self.stl_gen.save_stl(stl_mesh, temp_path):
                self.add_temp_file(temp_path)
                elapsed = time.time() - start_time
                logger.info(f"STL сгенерирован за {elapsed:.2f} секунд")
                return temp_path
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка генерации STL: {e}")
            traceback.print_exc()
            return None

# Интерфейс Gradio с улучшенными настройками маски
def create_gradio_interface():
    processor = MazeProcessor()
    
    with gr.Blocks(title="Улучшенный генератор лабиринтов с STL", theme=gr.themes.Soft()) as interface:
        maze_state = gr.State()
        mask_state = gr.State()
        
        gr.Markdown("""
        # 🧩 УЛУЧШЕННЫЙ ГЕНЕРАТОР ЛАБИРИНТОВ С STL ЭКСПОРТОМ
        Поддержка любых форматов изображений и улучшенное создание масок
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Основные настройки")
                
                shape_dropdown = gr.Dropdown(
                    choices=PREDEFINED_SHAPES,
                    value="Звезда",
                    label="Выберите форму",
                    interactive=True
                )
                
                use_custom = gr.Checkbox(
                    label="Использовать свое изображение",
                    value=False
                )
                
                image_input = gr.Image(
                    type="numpy",
                    label="Загрузите изображение (любой формат: JPG, PNG, BMP, SVG и т.д.)",
                    height=200,
                    visible=False
                )
                
                gr.Markdown("### 🎛️ Настройки маски (только для своих изображений)")
                
                with gr.Accordion("Расширенные настройки маски", open=False):
                    auto_invert = gr.Checkbox(
                        label="Автоматическая инверсия (если белый фон)",
                        value=True
                    )
                    
                    use_edge_detection = gr.Checkbox(
                        label="Использовать детектирование границ",
                        value=False,
                        info="Полезно для изображений со сложными границами"
                    )
                    
                    threshold_method = gr.Radio(
                        choices=["otsu", "adaptive", "triangle"],
                        value="otsu",
                        label="Метод бинаризации"
                    )
                
                gr.Markdown("### 🛠️ Улучшенные настройки для сложных изображений")
                
                with gr.Accordion("🛠️ Улучшенные настройки для сложных изображений", open=False):
                    advanced_method = gr.Radio(
                        choices=["auto", "adaptive", "edge_based", "grabcut"],
                        value="auto",
                        label="Метод создания маски"
                    )
                    
                    use_grabcut = gr.Checkbox(
                        label="Использовать GrabCut для уточнения",
                        value=True
                    )
                    
                    clahe_limit = gr.Slider(
                        minimum=1.0,
                        maximum=4.0,
                        value=2.0,
                        step=0.5,
                        label="Интенсивность улучшения контраста (CLAHE)"
                    )
                
                gr.Markdown("### 🎛️ Параметры лабиринта")
                
                image_size = gr.Slider(
                    minimum=200,
                    maximum=MAX_MAZE_SIZE,
                    value=min(DEFAULT_IMAGE_SIZE, MAX_MAZE_SIZE),
                    step=100,
                    label=f"Размер изображения (пиксели, макс: {MAX_MAZE_SIZE})"
                )
                
                wall_width_pixels = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Ширина стен (пиксели)"
                )
                
                generate_btn = gr.Button(
                    "🎲 Сгенерировать лабиринт",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("### 🖨️ Настройки 3D экспорта")
                
                wall_height_mm = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=DEFAULT_WALL_HEIGHT_MM,
                    step=1,
                    label="Высота стен (мм)"
                )
                
                wall_width_mm = gr.Slider(
                    minimum=0.5,
                    maximum=10,
                    value=DEFAULT_WALL_WIDTH_MM,
                    step=0.5,
                    label="Ширина стен (мм)"
                )
                
                base_height_mm = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=DEFAULT_BASE_HEIGHT_MM,
                    step=0.5,
                    label="Высота основания (мм)"
                )
                
                scale_factor = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Масштаб модели"
                )
                
                gr.Markdown("*Примечание: размер модели автоматически корректируется для 3D печати*")
                
                export_stl_btn = gr.Button(
                    "🔄 Сгенерировать STL",
                    variant="secondary"
                )
                
                gr.Markdown("### 💾 Скачать")
                
                download_png = gr.File(
                    label="Скачать PNG",
                    visible=False
                )
                
                download_stl = gr.File(
                    label="Скачать STL",
                    visible=False
                )
                
                gr.Markdown("### 📊 Статистика")
                stats_output = gr.JSON(
                    label="Статистика лабиринта",
                    value={}
                )
                
                gr.Markdown("""
                ### 🆕 Улучшения обработки изображений:
                - Поддержка любых форматов: JPG, PNG, BMP, GIF, TIFF, SVG
                - Автоматическая обработка прозрачности
                - Улучшение контраста (CLAHE)
                - Умная инверсия для светлого фона
                - Детектирование границ для сложных изображений
                - Заполнение внутренних областей
                - **НОВОЕ**: Расширенная обработка черных и разнотонных изображений
                - **НОВОЕ**: Алгоритм GrabCut для точной сегментации
                - **НОВОЕ**: Автоматический выбор лучшего метода
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Результат")
                output_image = gr.Image(
                    label="Сгенерированный лабиринт",
                    height=600,
                    type="numpy"
                )
                
                gr.Markdown("### 👁️ Предпросмотр маски")
                mask_preview = gr.Image(
                    label="Созданная маска",
                    height=300,
                    type="numpy",
                    visible=False
                )
        
        def toggle_image_input(use_custom_val):
            return gr.update(visible=use_custom_val), gr.update(interactive=not use_custom_val)
        
        def generate_maze(shape_name, uploaded_image, use_custom, img_size, wall_width,
                         auto_invert, use_edge_detection, threshold_method,
                         advanced_method, use_grabcut, clahe_limit):
            try:
                start_time = time.time()
                
                # Подготавливаем параметры маски
                mask_params = {
                    'auto_invert': auto_invert,
                    'use_edge_detection': use_edge_detection,
                    'threshold_method': threshold_method,
                    'advanced_method': advanced_method,
                    'use_grabcut': use_grabcut,
                    'clahe_limit': clahe_limit
                }
                
                result_image, maze, mask, stats = processor.process_maze(
                    shape_name, uploaded_image, use_custom, img_size, wall_width, mask_params
                )
                
                process_time = time.time() - start_time
                
                if "error" not in stats:
                    stats["Время генерации"] = f"{process_time:.2f} сек"
                    stats["Оптимизация"] = "Включена (объединение стен)"
                    stats["Метод маски"] = advanced_method
                
                png_path = processor.save_png(result_image) if result_image is not None else None
                
                # Создаем предпросмотр маски
                mask_preview_img = None
                if mask is not None:
                    mask_preview_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    mask_preview_img[mask] = [255, 255, 255]
                
                return (result_image, maze, mask, stats, png_path, None,
                       mask_preview_img, gr.update(visible=mask is not None))
                
            except Exception as e:
                error_msg = f"Ошибка: {str(e)}"
                logger.error(error_msg)
                error_image = processor._create_error_image(error_msg)
                return (error_image, None, None, {"error": error_msg}, None, None,
                       None, gr.update(visible=False))
        
        def generate_stl_file(maze, wall_height, wall_width, base_height, scale):
            try:
                if maze is None:
                    return None, gr.update(visible=False)
                
                stl_path = processor.generate_stl(
                    maze, wall_height, wall_width, base_height, scale
                )
                
                return stl_path, gr.update(visible=stl_path is not None)
                
            except Exception as e:
                error_msg = f"Ошибка генерации STL: {str(e)}"
                logger.error(error_msg)
                return None, gr.update(visible=False)
        
        def update_download_visibility(png_path, stl_path):
            return (
                gr.update(visible=png_path is not None, value=png_path),
                gr.update(visible=stl_path is not None, value=stl_path)
            )
        
        use_custom.change(
            fn=toggle_image_input,
            inputs=use_custom,
            outputs=[image_input, shape_dropdown]
        )
        
        generate_btn.click(
            fn=generate_maze,
            inputs=[shape_dropdown, image_input, use_custom, image_size, wall_width_pixels,
                   auto_invert, use_edge_detection, threshold_method,
                   advanced_method, use_grabcut, clahe_limit],
            outputs=[output_image, maze_state, mask_state, stats_output, download_png, 
                    download_stl, mask_preview, mask_preview]
        ).then(
            fn=update_download_visibility,
            inputs=[download_png, download_stl],
            outputs=[download_png, download_stl]
        )
        
        export_stl_btn.click(
            fn=generate_stl_file,
            inputs=[maze_state, wall_height_mm, wall_width_mm, base_height_mm, scale_factor],
            outputs=[download_stl, download_stl]
        )
    
    return interface

def main():
    print("=" * 70)
    print("🧩 УЛУЧШЕННЫЙ ГЕНЕРАТОР ЛАБИРИНТОВ С STL")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    
    print("\n✅ Улучшения:")
    print("   1. Поддержка всех форматов изображений (JPG, PNG, BMP, GIF, TIFF, SVG)")
    print("   2. Автоматическая обработка прозрачности")
    print("   3. Улучшение контраста (CLAHE)")
    print("   4. Умное определение фона и инверсия")
    print("   5. Детектирование границ для сложных изображений")
    print("   6. Заполнение внутренних областей контуров")
    print("   7. НОВОЕ: Расширенная обработка черных и разнотонных изображений")
    print("   8. НОВОЕ: Алгоритм GrabCut для точной сегментации")
    print("   9. НОВОЕ: Автоматический выбор лучшего метода")
    
    print("\n🚀 Запуск интерфейса...")
    print("   Откройте браузер: http://localhost:7860")
    print("   Для остановки: Ctrl+C\n")
    
    try:
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())