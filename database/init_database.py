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
    and_,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime, timedelta
import os
import enum
import sys

sys.path.append("..")
from settings import DB_CONFIG, DB_URL, SOURCE_METALS_CONFIG
from loguru import logger

Base = declarative_base()


class MetalType(enum.Enum):
    GOLD = "gold"
    COPPER = "copper"


class DataSource(enum.Enum):
    YFINANCE = "yfinance"


class MetalPrice(Base):
    __tablename__ = "metal_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metal_type = Column(Enum(MetalType), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    currency = Column(String(3), nullable=False, default="USD")
    source = Column(Enum(DataSource), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    is_checked = Column(
        Integer, nullable=False, default=1
    )  # 1=проверено, 0=не проверено
    is_market_closed = Column(
        Integer, nullable=False, default=0
    )  # 1=рынок закрыт, 0=нормально

    __table_args__ = (
        Index("idx_metal_prices_metal_timestamp", "metal_type", "timestamp"),
        Index("idx_metal_prices_source", "source"),
    )


class CollectorSchedule(Base):
    __tablename__ = "collector_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metal_type = Column(Enum(MetalType), nullable=False)
    source = Column(Enum(DataSource), nullable=False)
    interval_type = Column(String(10), nullable=False, default="hourly")
    is_active = Column(Integer, nullable=False, default=1)  # 1=активно, 0=неактивно
    last_run = Column(DateTime)
    last_triggered = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    @property
    def interval_minutes(self):
        """Возвращает интервал в минутах на основе interval_type"""
        if self.interval_type == "daily":
            return 24 * 60
        elif self.interval_type == "weekly":
            return 7 * 24 * 60
        else:
            return 60

    __table_args__ = (
        Index("idx_collector_schedules_metal_source", "metal_type", "source"),
    )


def drop_all_tables(engine):
    """Удаляет все таблицы в базе данных"""
    try:
        logger.info("Удаление всех существующих таблиц...")
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()

        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(text("SET FOREIGN_KEY_CHECKS=0;"))
                for table in all_tables:
                    connection.execute(text(f"DROP TABLE IF EXISTS {table};"))
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
    """Инициализация базы данных"""
    engine = create_engine(
        DB_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

    try:
        drop_all_tables(engine)
        logger.info("Все таблицы успешно удалены")
    except Exception as e:
        logger.error(f"Не удалось удалить таблицы: {e}")

    Base.metadata.create_all(engine)
    logger.info("Все таблицы успешно созданы")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        tables_count = len(Base.metadata.tables)
        existing_tables_count = 0

        for table_name in Base.metadata.tables.keys():
            if engine.dialect.has_table(engine.connect(), table_name):
                existing_tables_count += 1

        logger.info(
            f"База данных инициализирована: {existing_tables_count} из {tables_count} таблиц существует"
        )

        init_schedules(session, force_reset=force_reset_schedules)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка инициализации базы данных: {e}")
    finally:
        session.close()


def init_schedules(session, force_reset=False):
    """Инициализация расписаний сбора данных"""
    try:
        existing_count = session.query(CollectorSchedule).count()
        logger.info(f"В базе найдено {existing_count} расписаний")

        if existing_count > 0 and force_reset:
            session.query(CollectorSchedule).delete()
            session.commit()
            logger.info("Все существующие расписания удалены")
            create_default_schedules(session)
        elif existing_count == 0:
            create_default_schedules(session)

    except Exception as e:
        logger.error(f"Ошибка инициализации расписаний: {e}")
        try:
            engine = session.get_bind()
            with engine.connect() as connection:
                connection.execute(text("DROP TABLE IF EXISTS collector_schedules"))

            if hasattr(Base.metadata.tables, "collector_schedules"):
                Base.metadata.tables["collector_schedules"].create(engine)
                logger.info("Таблица CollectorSchedule успешно пересоздана")
                create_default_schedules(session)
            else:
                logger.error("Не удалось найти определение таблицы collector_schedules")
        except Exception as recreate_error:
            logger.error(f"Ошибка при пересоздании таблицы: {recreate_error}")


def create_default_schedules(session):
    """Создает расписания с часовым интервалом"""
    logger.info("Создание начальных расписаний с часовым интервалом")

    schedules_created = 0

    metal_name_to_enum = {
        "GOLD": MetalType.GOLD,
        "COPPER": MetalType.COPPER,
    }

    for source_name, allowed_metals in SOURCE_METALS_CONFIG.items():
        try:
            source = DataSource[source_name]

            for metal_name in allowed_metals:
                try:
                    metal_type = metal_name_to_enum[metal_name]
                    is_active = 1

                    schedule = CollectorSchedule(
                        metal_type=metal_type,
                        source=source,
                        interval_type="hourly",
                        is_active=is_active,
                    )

                    session.add(schedule)
                    schedules_created += 1
                    logger.debug(
                        f"Создано расписание для {metal_name} из {source_name}"
                    )
                except KeyError:
                    logger.warning(f"Неизвестный металл: {metal_name}, пропускаем")
                    continue
        except KeyError:
            logger.warning(f"Неизвестный источник данных: {source_name}, пропускаем")
            continue

    session.commit()
    logger.info(
        f"Создано {schedules_created} начальных расписаний с часовым интервалом на основе конфигурации"
    )


if __name__ == "__main__":
    logger.info("Запуск инициализации базы данных...")
    init_database()
    logger.info("Инициализация базы данных завершена!")
