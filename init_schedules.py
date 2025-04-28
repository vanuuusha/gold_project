#!/usr/bin/env python
"""
Инициализация расписаний для сбора данных.
Этот скрипт можно запустить отдельно, чтобы создать расписания для всех комбинаций
металлов и источников данных без пересоздания всей базы данных.
"""

import sys
import os
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Добавляем родительскую директорию в пути
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем нужные модули
from database.init_database import init_schedules
from settings import DB_URL, LOG_DIR

# Настройка логирования
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "init_schedules.log")

logger.remove()  # Удаляем стандартные обработчики
logger.add(
    log_file,
    rotation="1 MB",
    retention="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(sys.stderr, level="INFO")  # Вывод в консоль


def initialize_schedules():
    """Инициализация расписаний сбора данных для каждой комбинации металла и источника"""
    logger.info("Начало инициализации расписаний сбора данных")

    try:
        # Создаем подключение к базе данных
        engine = create_engine(DB_URL)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Спрашиваем пользователя, хочет ли он удалить существующие расписания
        force_reset = (
            input(
                "Пересоздать все расписания? Существующие будут удалены! (y/n): "
            ).lower()
            == "y"
        )

        # Инициализируем расписания
        init_schedules(session, force_reset=force_reset)
        logger.info("Инициализация расписаний завершена")

        session.close()
        return True

    except Exception as e:
        logger.error(f"Ошибка инициализации расписаний: {e}")
        return False


if __name__ == "__main__":
    success = initialize_schedules()
    if success:
        logger.info("Инициализация расписаний завершена успешно")
        print("Расписания успешно созданы!")
    else:
        logger.error("Не удалось инициализировать расписания")
        print("Ошибка при создании расписаний. Смотрите логи для деталей.")
    sys.exit(0 if success else 1)
