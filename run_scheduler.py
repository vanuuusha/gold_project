#!/usr/bin/env python3
"""
Скрипт для запуска планировщика задач обновления данных о ценах на металлы.
Выполняет проверку активных расписаний каждые 5 секунд и запускает задачи
на обновление данных, если настало время.
"""

import sys
import os
from loguru import logger
from database.data_collector import DataCollector


# Настройка логирования
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

if __name__ == "__main__":
    logger.info("Запуск планировщика задач обновления данных о ценах на металлы")

    try:
        # Создаем экземпляр коллектора данных
        collector = DataCollector()

        # Запускаем планировщик (будет работать бесконечно, проверяя задачи каждые 5 секунд)
        collector._run_task_scheduler()
    except KeyboardInterrupt:
        logger.warning("Планировщик остановлен пользователем (Ctrl+C)")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)
