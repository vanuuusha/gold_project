from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Index,
    Enum,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
import enum
import sys

sys.path.append("..")
from settings import DB_CONFIG, DB_URL
from loguru import logger

# Создание базового класса для декларативных моделей
Base = declarative_base()


# Определение перечисления типов металлов
class MetalType(enum.Enum):
    GOLD = "gold"
    COPPER = "copper"
    ALUMINUM = "aluminum"


# Определение перечисления источников данных
class DataSource(enum.Enum):
    YFINANCE = "yfinance"
    ALPHAVANTAGE_API = "alphavantage_api"


# Определение моделей
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default="active")
    role = Column(String(20), nullable=False, default="user")
    created_at = Column(DateTime, default=datetime.now)
    last_login = Column(DateTime)

    requests = relationship("UserRequest", back_populates="user")


class UserRequest(Base):
    __tablename__ = "user_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    request_type = Column(String(50), nullable=False)
    request_data = Column(Text)
    status = Column(String(20), nullable=False, default="pending")
    response_data = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime)

    user = relationship("User", back_populates="requests")

    __table_args__ = (Index("idx_user_requests_user_id", "user_id"),)


# Таблица для цен на металлы из разных источников
class MetalPrice(Base):
    __tablename__ = "metal_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metal_type = Column(Enum(MetalType), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(
        Float, nullable=True
    )  # Сделано nullable для периодов без данных, но отмеченных как проверенные
    high_price = Column(
        Float, nullable=True
    )  # Сделано nullable для периодов без данных, но отмеченных как проверенные
    low_price = Column(
        Float, nullable=True
    )  # Сделано nullable для периодов без данных, но отмеченных как проверенные
    close_price = Column(
        Float, nullable=True
    )  # Сделано nullable для периодов без данных, но отмеченных как проверенные
    currency = Column(String(3), nullable=False, default="USD")
    source = Column(Enum(DataSource), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    is_checked = Column(
        Integer, nullable=False, default=1
    )  # 1=проверено (даже если нет данных), 0=не проверено
    is_market_closed = Column(
        Integer, nullable=False, default=0
    )  # 1=рынок закрыт (выходные/праздники), 0=нормально

    __table_args__ = (
        Index("idx_metal_prices_metal_timestamp", "metal_type", "timestamp"),
        Index("idx_metal_prices_source", "source"),
    )


# Новая таблица для конфигурации расписаний сборщика данных
class CollectorSchedule(Base):
    __tablename__ = "collector_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metal_type = Column(Enum(MetalType), nullable=False)
    source = Column(Enum(DataSource), nullable=False)
    interval_type = Column(
        String(10), nullable=False, default="hourly"
    )  # hourly или daily
    is_active = Column(Integer, nullable=False, default=1)  # 1=активно, 0=неактивно
    last_run = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    @property
    def interval_minutes(self):
        """Возвращает интервал в минутах на основе interval_type"""
        if self.interval_type == "daily":
            return 24 * 60  # 24 часа в минутах
        elif self.interval_type == "weekly":
            return 7 * 24 * 60  # 7 дней в минутах
        else:
            return 60  # 1 час в минутах

    __table_args__ = (
        Index("idx_collector_schedules_metal_source", "metal_type", "source"),
    )


def drop_all_tables(engine):
    """Удаляет все таблицы в базе данных"""
    try:
        logger.info("Удаление всех существующих таблиц...")
        inspector = inspect(engine)

        # Получение списка всех таблиц в базе данных
        all_tables = inspector.get_table_names()

        # Удаление FOREIGN KEY ограничений перед удалением таблиц
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                # Отключение проверки ограничений
                connection.execute(text("SET FOREIGN_KEY_CHECKS=0;"))

                # Удаление каждой таблицы
                for table in all_tables:
                    connection.execute(text(f"DROP TABLE IF EXISTS {table};"))

                # Включение проверки ограничений
                connection.execute(text("SET FOREIGN_KEY_CHECKS=1;"))
                transaction.commit()
                logger.info(f"Успешно удалено {len(all_tables)} таблиц")
            except Exception as e:
                transaction.rollback()
                logger.error(f"Ошибка при удалении таблиц: {e}")
                raise
    except Exception as e:
        logger.error(f"Ошибка при удалении таблиц: {e}")
        raise


def init_database(force_reset_schedules=False):
    """
    Инициализация базы данных путем удаления и создания таблиц заново

    Args:
        force_reset_schedules (bool): Если True, существующие расписания будут удалены и созданы заново
    """
    # Создание соединения с базой данных используя SQLAlchemy
    engine = create_engine(
        DB_URL,
        echo=False,
        pool_pre_ping=True,  # Проверка соединения перед использованием
        pool_recycle=3600,  # Обновление соединений через 1 час
    )

    # Удаление существующих таблиц
    try:
        drop_all_tables(engine)
        logger.info("Все таблицы успешно удалены")
    except Exception as e:
        logger.error(f"Не удалось удалить таблицы: {e}")

    # Создание всех таблиц
    Base.metadata.create_all(engine)
    logger.info("Все таблицы успешно созданы")

    # Создание сессии
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Проверка успешного создания таблиц
        tables_count = len(Base.metadata.tables)
        existing_tables_count = 0

        for table_name in Base.metadata.tables.keys():
            if engine.dialect.has_table(engine.connect(), table_name):
                existing_tables_count += 1

        logger.info(
            f"База данных инициализирована: {existing_tables_count} из {tables_count} таблиц существует"
        )

        # Проверка и создание расписаний
        init_schedules(session, force_reset=force_reset_schedules)

        # Применение изменений
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка инициализации базы данных: {e}")
    finally:
        session.close()


def init_schedules(session, force_reset=False):
    """
    Инициализация расписаний сбора данных для каждой комбинации металла и источника

    Args:
        session: Сессия SQLAlchemy
        force_reset (bool): Если True, существующие расписания будут удалены и созданы заново
    """
    try:
        # Проверяем, есть ли уже расписания в таблице
        existing_count = session.query(CollectorSchedule).count()
        logger.info(f"В базе найдено {existing_count} расписаний")

        if existing_count > 0 and force_reset:
            # Удаляем все расписания
            session.query(CollectorSchedule).delete()
            session.commit()
            logger.info("Все существующие расписания удалены")
            # Теперь создаем заново
            create_default_schedules(session)
        elif existing_count == 0:
            # Если расписаний нет, создаем их
            create_default_schedules(session)

    except Exception as e:
        logger.error(f"Ошибка инициализации расписаний: {e}")
        # Пытаемся восстановить структуру таблицы, если возникла ошибка
        try:
            # Удаление и пересоздание таблицы
            engine = session.get_bind()
            with engine.connect() as connection:
                connection.execute(text("DROP TABLE IF EXISTS collector_schedules"))

            # Создаем таблицу заново
            if hasattr(Base.metadata.tables, "collector_schedules"):
                Base.metadata.tables["collector_schedules"].create(engine)
                logger.info("Таблица CollectorSchedule успешно пересоздана")

                # Создаем начальные расписания после пересоздания таблицы
                create_default_schedules(session)
            else:
                logger.error("Не удалось найти определение таблицы collector_schedules")
        except Exception as recreate_error:
            logger.error(f"Ошибка при пересоздании таблицы: {recreate_error}")


def create_default_schedules(session):
    """
    Создает начальные расписания с часовым интервалом для каждого металла и источника

    Args:
        session: Сессия SQLAlchemy
    """
    logger.info("Создание начальных расписаний с часовым интервалом")

    # Создаем расписания для каждой комбинации металла и источника
    schedules_created = 0

    for metal_type in MetalType:
        for source in DataSource:
            # Пропускаем некоторые невалидные комбинации
            if source == DataSource.ALPHAVANTAGE_API and metal_type == MetalType.COPPER:
                continue

            # Создаем расписание с часовым интервалом
            # По умолчанию активно только расписание для Gold из YFinance
            is_active = (
                1
                if (metal_type == MetalType.GOLD and source == DataSource.YFINANCE)
                else 0
            )

            schedule = CollectorSchedule(
                metal_type=metal_type,
                source=source,
                interval_type="hourly",
                is_active=is_active,
            )

            session.add(schedule)
            schedules_created += 1

    session.commit()
    logger.info(
        f"Создано {schedules_created} начальных расписаний с часовым интервалом"
    )


if __name__ == "__main__":
    # При прямом запуске этого файла инициализируем базу данных
    logger.info("Запуск инициализации базы данных...")
    init_database()
    logger.info("Инициализация базы данных завершена!")
