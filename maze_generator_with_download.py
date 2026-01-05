import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import random
import tempfile
import warnings
from datetime import datetime
import traceback
warnings.filterwarnings('ignore')

# ==================== КОНСТАНТЫ ====================
PREDEFINED_SHAPES = ["Сердце", "Звезда", "Круг", "Квадрат", "Треугольник", "Спираль"]
IMAGE_SIZE = 400
CELL_SIZE = 8  # Увеличил для лучшего качества
MODEL_PATH = "FastSAM-s.pt"

# ==================== УЛУЧШЕННЫЙ ГЕНЕРАТОР ЛАБИРИНТА ====================
class MazeGenerator:
    def __init__(self):
        self.directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
    
    def _validate_mask(self, mask):
        """Проверяет и очищает маску"""
        if mask is None or mask.size == 0:
            return None
        # Убеждаемся, что маска бинарная
        if mask.dtype != bool:
            mask = mask > 0
        # Удаляем мелкие шумы
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return None
        
        # Оставляем только самую большую компоненту
        sizes = ndimage.sum(mask, labeled, range(num_features + 1))
        largest_label = np.argmax(sizes[1:]) + 1
        mask_clean = labeled == largest_label
        
        # Заливаем мелкие дырки
        mask_clean = ndimage.binary_fill_holes(mask_clean)
        
        return mask_clean
    
    def _get_start_point(self, mask, grid_h, grid_w):
        """Находит лучшую стартовую точку в центре маски"""
        # Ищем центр масс маски
        from scipy import ndimage
        center_y, center_x = ndimage.center_of_mass(mask)
        center_y, center_x = int(center_y * grid_h / mask.shape[0]), int(center_x * grid_w / mask.shape[1])
        
        # Ищем ближайшую допустимую точку
        start_y, start_x = max(1, min(center_y, grid_h - 2)), max(1, min(center_x, grid_w - 2))
        
        # Проверяем окрестности если точка невалидна
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = start_y + dy, start_x + dx
                if 1 <= y < grid_h - 1 and 1 <= x < grid_w - 1:
                    if mask[int(y * mask.shape[0] / grid_h), int(x * mask.shape[1] / grid_w)]:
                        return (y, x)
        
        return (start_y, start_x)
    
    def generate_inside_mask(self, binary_mask, cell_size=CELL_SIZE):
        """Генерирует качественный лабиринт внутри бинарной маски"""
        try:
            from scipy import ndimage
            
            # Валидация и очистка маски
            binary_mask = self._validate_mask(binary_mask)
            if binary_mask is None or not np.any(binary_mask):
                raise ValueError("Маска пустая или невалидная")
            
            h, w = binary_mask.shape
            
            # Рассчитываем размер сетки для лабиринта
            grid_h = max(10, h // cell_size)
            grid_w = max(10, w // cell_size)
            
            # Создаем масштабированную маску с улучшенным качеством
            scale_y, scale_x = grid_h / h, grid_w / w
            
            # Используем режим 'constant' для сохранения формы
            scaled_mask = ndimage.zoom(
                binary_mask.astype(float), 
                (scale_y, scale_x), 
                order=0,  # Порядок 0 сохраняет четкие границы
                mode='constant',
                cval=0.0
            ) > 0.5
            
            # Улучшаем маску
            kernel = np.ones((3, 3), np.uint8)
            scaled_mask = ndimage.binary_erosion(scaled_mask, structure=kernel, iterations=1)
            scaled_mask = ndimage.binary_dilation(scaled_mask, structure=kernel, iterations=2)
            scaled_mask = ndimage.binary_fill_holes(scaled_mask)
            
            # Создаем лабиринт
            maze = np.ones((grid_h, grid_w), dtype=np.uint8)
            
            # Находим лучшую стартовую точку
            start = self._get_start_point(scaled_mask, grid_h, grid_w)
            
            # Проверяем, что стартовая точка внутри маски
            start_y, start_x = start
            if not (0 <= start_y < grid_h and 0 <= start_x < grid_w and scaled_mask[start_y, start_x]):
                # Ищем первую подходящую точку
                points = np.argwhere(scaled_mask)
                if len(points) == 0:
                    raise ValueError("Нет подходящих точек для старта")
                start = tuple(points[0])
            
            stack = [start]
            maze[start] = 0
            
            # Генерируем лабиринт с улучшенным алгоритмом
            while stack:
                y, x = stack[-1]
                random.shuffle(self.directions)
                moved = False
                
                # Проверяем все направления
                possible_moves = []
                for dy, dx in self.directions:
                    ny, nx = y + dy, x + dx
                    my, mx = y + dy // 2, x + dx // 2
                    
                    if (0 <= ny < grid_h and 0 <= nx < grid_w and
                        0 <= my < grid_h and 0 <= mx < grid_w and
                        scaled_mask[ny, nx] and maze[ny, nx] == 1):
                        possible_moves.append((dy, dx, ny, nx, my, mx))
                
                # Если есть возможные ходы, выбираем случайный
                if possible_moves:
                    dy, dx, ny, nx, my, mx = random.choice(possible_moves)
                    maze[my, mx] = 0
                    maze[ny, nx] = 0
                    stack.append((ny, nx))
                    moved = True
                
                if not moved:
                    stack.pop()
            
            # Масштабируем лабиринт обратно
            maze_fullsize = ndimage.zoom(
                maze, 
                (h / grid_h, w / grid_w), 
                order=0,
                mode='constant',
                cval=1.0
            )
            
            # Обрезаем до исходного размера
            maze_fullsize = maze_fullsize[:h, :w]
            
            # Применяем исходную маску
            maze_fullsize = np.where(binary_mask, maze_fullsize, 1)
            
            # Улучшаем качество границ
            maze_fullsize = ndimage.binary_dilation(maze_fullsize == 0, iterations=1).astype(np.uint8)
            maze_fullsize = ndimage.binary_erosion(maze_fullsize == 1, iterations=1).astype(np.uint8)
            
            return maze_fullsize
            
        except Exception as e:
            print(f"Ошибка генерации лабиринта: {e}")
            traceback.print_exc()
            raise

# ==================== УЛУЧШЕННАЯ БАЗА ФОРМ ====================
class ShapeDatabase:
    @staticmethod
    def create_heart_mask(width=IMAGE_SIZE, height=IMAGE_SIZE):
        """Создает маску сердца с улучшенным качеством"""
        mask = np.zeros((height, width), dtype=bool)
        center_x, center_y = width // 2, height // 2
        size = min(width, height) // 3
        
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        
        # Уравнение сердца
        heart_eq = (x**2 + (1.2*y - np.sqrt(np.abs(x)))**2 - size**2) < 0
        
        mask[heart_eq] = True
        
        # Сглаживание
        from scipy import ndimage
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
        mask = ndimage.binary_fill_holes(mask)
        
        return mask
    
    @staticmethod
    def create_star_mask(width=IMAGE_SIZE, height=IMAGE_SIZE, points=5):
        """Создает маску звезды с улучшенным качеством"""
        mask = np.zeros((height, width), dtype=bool)
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2.5
        
        # Создаем полярную сетку
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Формула для звезды
        star_r = radius * (1 + 0.5 * np.sin(points * theta)) / (1 + 0.5)
        
        mask[r < star_r] = True
        
        # Улучшаем качество
        from scipy import ndimage
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_fill_holes(mask)
        
        return mask
    
    @staticmethod
    def create_circle_mask(width=IMAGE_SIZE, height=IMAGE_SIZE):
        mask = np.zeros((height, width), dtype=bool)
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        mask[x**2 + y**2 <= radius**2] = True
        
        return mask
    
    @staticmethod
    def create_square_mask(width=IMAGE_SIZE, height=IMAGE_SIZE):
        mask = np.zeros((height, width), dtype=bool)
        margin = min(width, height) // 4
        mask[margin:height-margin, margin:width-margin] = True
        return mask
    
    @staticmethod
    def create_triangle_mask(width=IMAGE_SIZE, height=IMAGE_SIZE):
        mask = np.zeros((height, width), dtype=bool)
        
        # Вершины треугольника
        vertices = np.array([
            [width // 2, height // 4],           # Верх
            [width // 4, 3 * height // 4],       # Левый низ
            [3 * width // 4, 3 * height // 4]    # Правый низ
        ])
        
        # Создаем сетку точек
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        points = np.stack([x.ravel(), y.ravel()], axis=1)
        
        # Проверяем, находится ли точка внутри треугольника
        def point_in_triangle(pt, v1, v2, v3):
            d1 = np.sign((pt[0] - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (pt[1] - v2[1]))
            d2 = np.sign((pt[0] - v3[0]) * (v2[1] - v3[1]) - (v2[0] - v3[0]) * (pt[1] - v3[1]))
            d3 = np.sign((pt[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (pt[1] - v1[1]))
            return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)
        
        # Применяем проверку ко всем точкам
        for i, point in enumerate(points):
            if point_in_triangle(point, vertices[0], vertices[1], vertices[2]):
                mask[point[1], point[0]] = True
        
        # Улучшаем качество
        from scipy import ndimage
        mask = ndimage.binary_fill_holes(mask)
        
        return mask
    
    @staticmethod
    def create_spiral_mask(width=IMAGE_SIZE, height=IMAGE_SIZE):
        """Создает маску спирали с улучшенным качеством"""
        mask = np.zeros((height, width), dtype=bool)
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2 - 20
        
        # Создаем спираль с несколькими витками
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Уравнение спирали
        spiral_r = 10 + (max_radius / (4 * np.pi)) * (theta + 4 * np.pi)
        
        # Толщина линии
        thickness = 8
        mask[np.abs(r - spiral_r) < thickness] = True
        
        # Улучшаем качество
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_fill_holes(mask)
        
        return mask
    
    @classmethod
    def get_mask(cls, shape_name):
        shape_name = shape_name.lower()
        if 'сердц' in shape_name or 'heart' in shape_name:
            return cls.create_heart_mask()
        elif 'звезд' in shape_name or 'star' in shape_name:
            return cls.create_star_mask()
        elif 'круг' in shape_name or 'circle' in shape_name:
            return cls.create_circle_mask()
        elif 'квадрат' in shape_name or 'square' in shape_name:
            return cls.create_square_mask()
        elif 'треуголь' in shape_name or 'triangle' in shape_name:
            return cls.create_triangle_mask()
        elif 'спирал' in shape_name or 'spiral' in shape_name:
            return cls.create_spiral_mask()
        return cls.create_heart_mask()

# ==================== УЛУЧШЕННЫЙ FASTSAM ОБРАБОТЧИК ====================
class FastSAMProcessor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Загружает модель FastSAM"""
        try:
            from ultralytics import FastSAM
            if Path(self.model_path).exists():
                self.model = FastSAM(self.model_path)
                print(f"✅ FastSAM модель загружена из {self.model_path}")
            else:
                print(f"⚠️ Файл модели не найден: {self.model_path}")
                print("📥 Скачайте модель FastSAM-s.pt:")
                print("https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.1/FastSAM-s.pt")
                self.model = None
        except ImportError:
            print("❌ Установите ultralytics: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
    
    def process_image(self, image):
        """Обрабатывает изображение и возвращает маску"""
        if self.model is None:
            print("⚠️ Модель FastSAM не загружена!")
            return None
        
        if image is None:
            print("⚠️ Изображение не предоставлено!")
            return None
        
        try:
            # Конвертируем изображение в RGB
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:  # RGB или BGR
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:  # Grayscale
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Изменяем размер для обработки
            h, w = image_rgb.shape[:2]
            target_size = max(640, min(h, w, 1024))
            
            # Обработка через FastSAM
            results = self.model(
                image_rgb, 
                device="cpu", 
                imgsz=target_size,
                conf=0.25,  # Более низкий порог для лучшего обнаружения
                iou=0.7,
                retina_masks=True
            )
            
            masks = results[0].masks
            if masks is None or len(masks) == 0:
                print("⚠️ FastSAM не нашел объектов на изображении")
                return None
            
            # Выбираем лучшую маску
            mask_data = masks.data.cpu().numpy()
            
            # Для каждой маски считаем площадь и качество
            best_mask_idx = 0
            best_score = -1
            
            for i, mask in enumerate(mask_data):
                # Площадь маски
                area = mask.sum()
                # Координаты ограничивающей рамки
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, 0)
                xmin, xmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, 0)
                bbox_area = (ymax - ymin) * (xmax - xmin)
                
                # Счет = площадь * компактность
                compactness = area / bbox_area if bbox_area > 0 else 0
                score = area * compactness
                
                if score > best_score:
                    best_score = score
                    best_mask_idx = i
            
            binary_mask = mask_data[best_mask_idx] > 0
            
            # Изменяем размер маски к исходному размеру изображения
            if binary_mask.shape != (h, w):
                binary_mask = cv2.resize(
                    binary_mask.astype(np.uint8), 
                    (w, h), 
                    interpolation=cv2.INTER_NEAREST
                ) > 0
            
            # Улучшаем качество маски
            kernel = np.ones((7, 7), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Заполняем дыры
            from scipy import ndimage
            binary_mask = ndimage.binary_fill_holes(binary_mask)
            
            print(f"✅ Маска успешно создана, размер: {binary_mask.shape}, площадь: {binary_mask.sum()} пикселей")
            return binary_mask > 0
            
        except Exception as e:
            print(f"❌ Ошибка обработки изображения: {e}")
            traceback.print_exc()
            return None

# ==================== ОСНОВНОЙ ПРОЦЕССОР ====================
class MazeApp:
    def __init__(self):
        self.maze_gen = MazeGenerator()
        self.shape_db = ShapeDatabase()
        self.sam_processor = FastSAMProcessor()
        self.setup_colors()
    
    def setup_colors(self):
        self.COLORS = {
            'wall': [30, 30, 30],
            'path': [240, 240, 240],
            'start': [76, 175, 80],
            'end': [244, 67, 54],
            'highlight': [33, 150, 243, 128]
        }
    
    def process(self, shape_name, uploaded_image, use_custom_image):
        """Основная функция обработки"""
        try:
            print(f"\n{'='*50}")
            print(f"🔄 Начало обработки")
            print(f"{'='*50}")
            
            if use_custom_image and uploaded_image is not None:
                print(f"📷 Режим: пользовательское изображение")
                print(f"📐 Размер изображения: {uploaded_image.shape}")
                
                binary_mask = self.sam_processor.process_image(uploaded_image)
                
                if binary_mask is None or not np.any(binary_mask):
                    print("⚠️ FastSAM не смог обработать изображение, использую форму по умолчанию")
                    binary_mask = self.shape_db.get_mask(shape_name)
                else:
                    print(f"✅ Маска создана успешно")
            else:
                print(f"📐 Режим: предопределенная форма '{shape_name}'")
                binary_mask = self.shape_db.get_mask(shape_name)
                print(f"✅ Форма создана успешно")
            
            if binary_mask is None:
                print("❌ Ошибка: маска не создана")
                return self.create_error_image("Ошибка создания маски"), None
            
            # Изменяем размер маски к стандартному размеру
            binary_mask = cv2.resize(
                binary_mask.astype(np.uint8), 
                (IMAGE_SIZE, IMAGE_SIZE), 
                interpolation=cv2.INTER_NEAREST
            ) > 0
            
            print(f"🌀 Генерация лабиринта...")
            print(f"📊 Размер маски: {binary_mask.shape}")
            print(f"📈 Площадь маски: {binary_mask.sum()} пикселей ({binary_mask.sum()/(IMAGE_SIZE*IMAGE_SIZE)*100:.1f}%)")
            
            maze = self.maze_gen.generate_inside_mask(binary_mask, CELL_SIZE)
            
            print(f"✅ Лабиринт сгенерирован успешно")
            print(f"📊 Размер лабиринта: {maze.shape}")
            print(f"📈 Проходов/стен: {np.sum(maze==0)}/{np.sum(maze==1)}")
            
            result = self.visualize_maze(maze, binary_mask)
            
            # Сохранение для скачивания
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_shape_name = "".join(c for c in shape_name if c.isalnum() or c in (' ', '_')).rstrip()
            filename = f"maze_{safe_shape_name}_{timestamp}.png"
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="maze_")
            temp_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(temp_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"💾 Лабиринт сохранен: {temp_path}")
            print(f"✅ Обработка завершена успешно!")
            print(f"{'='*50}\n")
            
            return result, temp_path
            
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            traceback.print_exc()
            return self.create_error_image(str(e)), None
    
    def visualize_maze(self, maze, mask):
        """Визуализирует лабиринт с цветовым кодированием"""
        h, w = maze.shape
        
        # Создаем цветное изображение
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Стены
        colored[maze == 1] = self.COLORS['wall']
        # Проходы
        colored[maze == 0] = self.COLORS['path']
        
        # Находим и отмечаем старт и финиш
        colored = self.add_start_end(colored, maze, mask)
        
        # Подсвечиваем границы формы
        colored = self.highlight_shape(colored, mask)
        
        return colored
    
    def add_start_end(self, image, maze, mask):
        """Добавляет старт и финиш в лучшие позиции"""
        h, w = maze.shape
        
        # Ищем точки внутри маски
        points = np.argwhere(mask & (maze == 0))
        if len(points) < 2:
            return image
        
        # Старт - точка с минимальным расстоянием до центра
        center = np.array([h//2, w//2])
        distances = np.linalg.norm(points - center, axis=1)
        start_idx = np.argmin(distances)
        start = tuple(points[start_idx])
        
        # Финиш - точка максимально удаленная от старта
        start_point = np.array(start)
        distances_to_start = np.linalg.norm(points - start_point, axis=1)
        end_idx = np.argmax(distances_to_start)
        end = tuple(points[end_idx])
        
        # Рисуем старт (зеленый)
        y, x = start
        radius = max(3, min(h, w) // 50)
        cv2.circle(image, (x, y), radius, self.COLORS['start'][:3], -1)
        cv2.circle(image, (x, y), radius, (255, 255, 255), 1)
        
        # Рисуем финиш (красный)
        y, x = end
        cv2.circle(image, (x, y), radius, self.COLORS['end'][:3], -1)
        cv2.circle(image, (x, y), radius, (255, 255, 255), 1)
        
        return image
    
    def highlight_shape(self, image, mask):
        """Подсвечивает границы формы"""
        from scipy import ndimage
        
        # Находим контур
        contour = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
        
        # Рисуем контур синим цветом
        contour_coords = np.where(contour)
        for y, x in zip(*contour_coords):
            # Плавное смешивание с текущим цветом
            alpha = 0.3
            current_color = image[y, x].astype(float)
            highlight_color = np.array(self.COLORS['highlight'][:3])
            image[y, x] = (current_color * (1 - alpha) + highlight_color * alpha).astype(np.uint8)
        
        return image
    
    def create_error_image(self, message):
        """Создает изображение с сообщением об ошибке"""
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img[:] = [40, 40, 60]  # Темный фон
        
        try:
            # Добавляем текст ошибки
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "ОШИБКА"
            text_size = cv2.getTextSize(text, font, 1.5, 2)[0]
            text_x = (IMAGE_SIZE - text_size[0]) // 2
            text_y = IMAGE_SIZE // 2 - 30
            cv2.putText(img, text, (text_x, text_y), font, 1.5, (255, 100, 100), 2, cv2.LINE_AA)
            
            # Добавляем сообщение
            if len(message) > 40:
                message = message[:37] + "..."
            msg_size = cv2.getTextSize(message, font, 0.7, 1)[0]
            msg_x = (IMAGE_SIZE - msg_size[0]) // 2
            msg_y = IMAGE_SIZE // 2 + 30
            cv2.putText(img, message, (msg_x, msg_y), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
            
        except:
            pass
        
        return img

# ==================== GRADIO ИНТЕРФЕЙС ====================
def create_interface():
    """Создает веб-интерфейс"""
    app = MazeApp()

    with gr.Blocks(title="Генератор лабиринтов с FastSAM", theme=gr.themes.Soft()) as interface:
        gr.Markdown(""" 
        # 🧩 Генератор лабиринтов в произвольной форме
        ### Создавайте красивые лабиринты внутри любых форм!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Настройки")
                
                shape_dropdown = gr.Dropdown(
                    choices=PREDEFINED_SHAPES,
                    value="Сердце",
                    label="📐 Выберите форму",
                    interactive=True
                )
                
                use_custom = gr.Checkbox(
                    label="🖼️ Использовать свое изображение",
                    value=False,
                    interactive=True
                )
                
                image_input = gr.Image(
                    type="numpy",
                    label="📤 Загрузите изображение",
                    height=200,
                    visible=False
                )
                
                gr.Markdown("### 🎨 Цветовая схема")
                gr.Markdown("""
                - 🟩 **Зеленый** - старт лабиринта
                - 🟥 **Красный** - финиш лабиринта
                - 🔵 **Синий** - границы формы
                - ⬛ **Темный** - стены лабиринта
                - ⬜ **Светлый** - проходы лабиринта
                """)
                
                generate_btn = gr.Button(
                    "🎲 Сгенерировать лабиринт",
                    variant="primary",
                    size="lg"
                )
                
                download_btn = gr.File(
                    label="💾 Скачать лабиринт (PNG)",
                    visible=False
                )
                
                gr.Markdown("""
                ---
                ### 💡 Советы:
                1. Для лучших результатов используйте изображения с четким объектом на контрастном фоне
                2. Изображения должны быть не менее 300x300 пикселей
                3. Сложные формы могут требовать больше времени для обработки
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Результат")
                output_image = gr.Image(
                    label="Сгенерированный лабиринт", 
                    height=500,
                    type="numpy"
                )
                
                with gr.Accordion("📊 Информация о обработке", open=False):
                    info_text = gr.Textbox(
                        label="Лог обработки",
                        lines=5,
                        interactive=False
                    )
        
        # Функции
        def toggle_visibility(use_custom_val):
            return {
                shape_dropdown: gr.update(interactive=not use_custom_val),
                image_input: gr.update(visible=use_custom_val)
            }
        
        def process_wrapper(shape_name, uploaded_image, use_custom_image):
            result, file_path = app.process(shape_name, uploaded_image, use_custom_image)
            return result, file_path, f"Обработка завершена. Файл сохранен: {file_path if file_path else 'Ошибка'}"
        
        # Обработчики событий
        use_custom.change(
            fn=toggle_visibility,
            inputs=use_custom,
            outputs=[shape_dropdown, image_input]
        )
        
        generate_btn.click(
            fn=process_wrapper,
            inputs=[shape_dropdown, image_input, use_custom],
            outputs=[output_image, download_btn, info_text]
        ).then(
            fn=lambda file_path: gr.update(visible=file_path is not None),
            inputs=[download_btn],
            outputs=[download_btn]
        )
        
        # Загрузка интерфейса
        interface.load(
            fn=lambda: None,
            inputs=None,
            outputs=None,
            _js="""
            () => {
                console.log('✅ Интерфейс загружен!');
                alert('Готов к работе! Выберите форму или загрузите изображение.');
            }
            """
        )

    return interface

def main():
    """
    Основная функция для запуска приложения
    """
    print("=" * 60)
    print("🧩 Генератор лабиринтов с FastSAM - УЛУЧШЕННАЯ ВЕРСИЯ")
    print("=" * 60)
    
    # Проверка зависимостей
    try:
        import scipy
        print(f"✅ SciPy установлена: {scipy.__version__}")
    except ImportError:
        print("❌ SciPy не установлена! Установите: pip install scipy")
        return
    
    try:
        import ultralytics
        print(f"✅ Ultralytics установлен: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics не установлен! Установите: pip install ultralytics")
        print("   Это необходимо для работы FastSAM")
    
    # Проверка модели FastSAM
    if not Path(MODEL_PATH).exists():
        print(f"\n⚠️  Внимание: файл модели '{MODEL_PATH}' не найден!")
        print("\n📥  Чтобы исправить это:")
        print("   1. Скачайте модель FastSAM-s.pt по ссылке:")
        print("      https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.1/FastSAM-s.pt")
        print("   2. Поместите скачанный файл в папку с этим скриптом.")
        print(f"      Текущая папка: {Path.cwd()}")
        print("\n⚠️  Приложение будет работать только с предопределенными формами.")
        print("   Для обработки изображений требуется модель FastSAM.\n")
    
    print("\n🚀  Запуск веб-интерфейса...")
    print("   Откройте браузер и перейдите по адресу: http://localhost:7860")
    print("   Для остановки сервера нажмите Ctrl+C\n")
    
    # Запуск интерфейса
    try:
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=False
        )
    except Exception as e:
        print(f"❌  Не удалось запустить приложение. Ошибка: {e}")
        print("\n🔧  Возможные причины:")
        print("   - Порт 7860 занят (попробуйте другой порт)")
        print("   - Проблема с установкой Gradio")
        print("   - Ошибка в коде интерфейса")

# ==================== ТОЧКА ВХОДА ====================
if __name__ == "__main__":
    main()