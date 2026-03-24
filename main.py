"""
Прототип системы детекции уборки столиков по видео
Использует YOLOv8 для детекции людей и анализирует события для выбранного столика
Запуск: python main.py --video video1.mp4
"""

import argparse
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import time

class TableCleaningDetector:
    def __init__(self, video_path, roi=None, skip_frames=15, confidence_threshold=0.3):
        """
        Инициализация детектора
        
        Args:
            video_path: путь к видеофайлу
            roi: область интереса (x, y, w, h) - если None, выбирается вручную
            skip_frames: обрабатывать каждый N-й кадр для ускорения
            confidence_threshold: порог уверенности YOLO (снижен для лучшей детекции сидящих)
        """
        self.video_path = video_path
        self.roi = roi
        self.skip_frames = skip_frames
        self.confidence_threshold = confidence_threshold
        
        # Загружаем предобученную YOLOv8 nano на CPU
        self.model = YOLO('yolov8n.pt')
        self.model.to('cpu')
        
        # Состояния столика
        self.STATE_EMPTY = "empty"
        self.STATE_OCCUPIED = "occupied"
        self.current_state = self.STATE_EMPTY
        
        # Данные для аналитики
        self.events = []
        self.empty_start_time = None
        self.cleaning_times = []
        
        # Таймеры для защиты от ложных срабатываний
        self.last_seen_time = 0.0
        self.empty_threshold_seconds = 3.0  # Стол пуст, если человека нет > 3 секунд
        
    def select_roi(self, frame):
        """Выбор области столика вручную на первом кадре с подгонкой под размер экрана"""
        print("\nПожалуйста, выделите область столика ВМЕСТЕ СО СТУЛЬЯМИ и нажмите ENTER или SPACE")
        print("Для отмены нажмите ESC")
        
        # Создаем окно, размер которого можно менять
        cv2.namedWindow("Select Table ROI", cv2.WINDOW_NORMAL)
        
        # Вычисляем пропорции и подгоняем размер окна под экран (макс ширина 1280px)
        height, width = frame.shape[:2]
        max_width = 1280 
        
        if width > max_width:
            scaled_height = int(height * (max_width / width))
            cv2.resizeWindow("Select Table ROI", max_width, scaled_height)
        else:
            cv2.resizeWindow("Select Table ROI", width, height)
            
        roi = cv2.selectROI("Select Table ROI", frame, False, False)
        cv2.destroyWindow("Select Table ROI")
        
        if roi == (0, 0, 0, 0):
            raise ValueError("Область столика не выбрана!")
        
        return roi
    
    def is_person_in_roi(self, frame):
        """
        Проверяет, есть ли человек в области столика.
        Анализирует весь кадр для сохранения контекста, затем проверяет пересечение координат.
        """
        if self.roi is None:
            return False
        
        rx, ry, rw, rh = self.roi
        
        # Детекция людей на всем кадре (class 0 = person)
        results = self.model(frame, classes=[0], conf=self.confidence_threshold, 
                             verbose=False, device='cpu')
        
        for result in results:
            for box in result.boxes.xyxy:  # Получаем координаты [x1, y1, x2, y2]
                px1, py1, px2, py2 = map(int, box[:4])
                
                # Проверяем пересечение прямоугольника человека и области стола (ROI)
                if px1 < rx + rw and px2 > rx and py1 < ry + rh and py2 > ry:
                    return True
                    
        return False
    
    def update_state(self, timestamp, has_person):
        """Обновляет состояние столика и фиксирует события в лог"""
        # Логика сглаживания (debouncing)
        if has_person:
            self.last_seen_time = timestamp
            suggested_state = self.STATE_OCCUPIED
        else:
            # Считаем стол пустым, только если человека не было видно дольше порога
            if (timestamp - self.last_seen_time) > self.empty_threshold_seconds:
                suggested_state = self.STATE_EMPTY
            else:
                suggested_state = self.STATE_OCCUPIED
        
        # Если состояние действительно изменилось
        if suggested_state != self.current_state:
            old_state = self.current_state
            self.current_state = suggested_state
            
            event = {
                'timestamp': round(timestamp, 2),
                'from_state': old_state,
                'to_state': self.current_state,
                'event_type': '',
                'cleaning_time_sec': None
            }
            
            # Определяем тип события и считаем аналитику
            if self.current_state == self.STATE_OCCUPIED and old_state == self.STATE_EMPTY:
                event['event_type'] = 'approach'
                
                # Считаем время между уходом предыдущего и подходом нового
                if self.empty_start_time is not None:
                    cleaning_time = timestamp - self.empty_start_time
                    self.cleaning_times.append(cleaning_time)
                    event['cleaning_time_sec'] = round(cleaning_time, 2)
                    print(f"-> Зафиксирован подход! Время пустования/уборки стола: {cleaning_time:.2f} сек.")
                    
            elif self.current_state == self.STATE_EMPTY and old_state == self.STATE_OCCUPIED:
                event['event_type'] = 'leave'
                self.empty_start_time = timestamp
                print(f"-> Гость ушел. Стол освободился.")
            
            self.events.append(event)
            print(f"[{timestamp:.2f}с] Статус изменен на: {self.current_state.upper()}")
    
    def process_video(self):
        """Основной цикл обработки видео"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать первый кадр")
        
        if self.roi is None:
            self.roi = self.select_roi(first_frame)
            
        print(f"\nВыбран столик: {self.roi}")
        print("Начинаю обработку видео...\n")
        
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            timestamp = frame_count / fps
            
            # Детекция только на каждом N-ом кадре (для скорости)
            if frame_count % self.skip_frames == 0:
                processed_frames += 1
                has_person = self.is_person_in_roi(frame)
                self.update_state(timestamp, has_person)
                
                if processed_frames % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Прогресс: {progress:.1f}%")
            
            # Отрисовка на каждом кадре для плавности видео
            x, y, w, h = self.roi
            if self.current_state == self.STATE_EMPTY:
                color = (0, 255, 0)  # Зеленый - пусто
                status_text = "EMPTY"
            else:
                color = (0, 0, 255)  # Красный - занято
                status_text = "OCCUPIED"
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"Table: {status_text}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Time: {timestamp:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nОбработка завершена за {time.time() - start_time:.1f} секунд.")
        return output_path
        
    def generate_report(self):
        """Генерирует DataFrame, считает статистику и сохраняет результаты"""
        if not self.events:
            print("Событий не зарегистрировано.")
            return None
            
        # 1. Создание Pandas DataFrame согласно заданию
        events_df = pd.DataFrame(self.events)
        
        # 2. Сохранение DataFrame в CSV файл
        csv_path = "events_log.csv"
        events_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 3. Базовая статистика
        print("\n" + "="*50)
        print("ОТЧЕТ ПО АНАЛИТИКЕ")
        print("="*50)
        print("\nВременная шкала событий (Pandas DataFrame):")
        print(events_df.to_string())
        
        report_text = "ОТЧЕТ ПО ДЕТЕКЦИИ УБОРКИ СТОЛИКОВ\n" + "="*50 + "\n\n"
        report_text += "СОБЫТИЯ:\n" + events_df.to_string() + "\n\nСТАТИСТИКА:\n"
        
        if self.cleaning_times:
            mean_time = np.mean(self.cleaning_times)
            stats = (
                f"Количество циклов (уход -> подход): {len(self.cleaning_times)}\n"
                f"Среднее время простоя/уборки стола: {mean_time:.2f} секунд\n"
                f"Минимальное время: {np.min(self.cleaning_times):.2f} секунд\n"
                f"Максимальное время: {np.max(self.cleaning_times):.2f} секунд\n"
            )
            print("\n" + stats)
            report_text += stats
        else:
            msg = "Нет данных о полных циклах уборки (не зафиксирован уход + последующий подход)\n"
            print("\n" + msg)
            report_text += msg
            
        # 4. Сохранение текстового отчета
        report_path = "report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        return events_df

def main():
    parser = argparse.ArgumentParser(description='Детекция уборки столиков по видео')
    parser.add_argument('--video', type=str, required=True, help='Путь к видеофайлу')
    parser.add_argument('--roi', type=str, help='Область интереса в формате "x,y,w,h" (опционально)')
    parser.add_argument('--skip-frames', type=int, default=10, 
                        help='Обрабатывать каждый N-й кадр (по умолчанию: 10)')
    
    args = parser.parse_args()
    
    roi = None
    if args.roi:
        roi = tuple(map(int, args.roi.split(',')))
        
    detector = TableCleaningDetector(args.video, roi, skip_frames=args.skip_frames)
    
    # Запуск пайплайна
    output_video = detector.process_video()
    events_df = detector.generate_report()
    
    print("\n" + "*"*50)
    print("ИТОГОВЫЕ ФАЙЛЫ ПРОЕКТА:")
    print(f"1. Видео с визуализацией:  {output_video}")
    print(f"2. Таблица Pandas (CSV):   events_log.csv")
    print(f"3. Текстовый отчет:        report.txt")
    print("*"*50)

if __name__ == "__main__":
    main()
