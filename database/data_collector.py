import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import and_, func
from loguru import logger
from sqlalchemy.orm import sessionmaker
from database.database_client import DatabaseClient
from database.init_database import MetalPrice, MetalType, DataSource, CollectorSchedule
import sys
import time
import threading


class DataCollector:
    """Класс для сбора данных о ценах на металлы из различных источников."""

    def __init__(self):
        self.db_client = DatabaseClient()
        self.engine = self.db_client.cnx
        self.Session = sessionmaker(bind=self.engine)
        self.running_tasks = {}
        self._scheduler_thread = None

    def get_metal_prices_yfinance(
        self, metal_type: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Получает цены металлов за период"""
        start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=1)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        end_date = end_date.strftime("%Y-%m-%d")
        yf_ticker = yf.Ticker(metal_type)
        yf_data = yf_ticker.history(start=start_date, end=end_date, interval="1h")
        return yf_data

    def update_missing_metal_prices(
        self, metal_name: str, default_start_date: str = "2024-01-01"
    ) -> int:
        """Обновляет недостающие данные для указанного металла"""
        mappings = {
            "gold": "GC=F",
            "copper": "HG=F",
        }

        if metal_name.lower() not in mappings:
            raise ValueError(
                f"Неподдерживаемый тип металла: {metal_name}. Поддерживаются только: {', '.join(mappings.keys())}"
            )

        ticker = mappings[metal_name.lower()]
        end_date = datetime.now().strftime("%Y-%m-%d")
        metal_type = (
            MetalType.GOLD if metal_name.lower() == "gold" else MetalType.COPPER
        )

        session = self.Session()

        try:
            latest_record = (
                session.query(MetalPrice)
                .filter(
                    and_(
                        MetalPrice.metal_type == metal_type,
                        MetalPrice.source == DataSource.YFINANCE,
                    )
                )
                .order_by(MetalPrice.timestamp.desc())
                .first()
            )

            if latest_record:
                start_date = latest_record.timestamp.strftime("%Y-%m-%d")
                logger.info(
                    f"Найдена последняя запись для {metal_name} от {latest_record.timestamp}. Используем {start_date} как начальную дату"
                )
            else:
                start_date = default_start_date
                logger.info(
                    f"Записи для {metal_name} не найдены. Используем дату по умолчанию: {start_date}"
                )

            logger.info(
                f"Начинаем обновление данных для {metal_name} ({ticker}) с {start_date} по {end_date}"
            )

            yf_data = self.get_metal_prices_yfinance(ticker, start_date, end_date)
            if yf_data.empty:
                logger.warning(
                    f"Не удалось получить данные для {metal_name} из YFinance"
                )
                return 0

            logger.info(f"Получено {len(yf_data)} строк данных из YFinance")

            existing_records = (
                session.query(MetalPrice.timestamp)
                .filter(
                    and_(
                        MetalPrice.metal_type == metal_type,
                        MetalPrice.source == DataSource.YFINANCE,
                        MetalPrice.timestamp >= start_date,
                        MetalPrice.timestamp <= end_date,
                    )
                )
                .all()
            )

            existing_timestamps = {record.timestamp for record in existing_records}
            records_added = 0

            for timestamp, row in yf_data.iterrows():
                if timestamp not in existing_timestamps:
                    duplicate_check = (
                        session.query(MetalPrice)
                        .filter(
                            and_(
                                MetalPrice.metal_type == metal_type,
                                MetalPrice.source == DataSource.YFINANCE,
                                MetalPrice.timestamp == timestamp,
                            )
                        )
                        .first()
                    )

                    if not duplicate_check:
                        new_price = MetalPrice(
                            metal_type=metal_type,
                            timestamp=timestamp,
                            open_price=row.get("Open"),
                            high_price=row.get("High"),
                            low_price=row.get("Low"),
                            close_price=row.get("Close"),
                            currency="USD",
                            source=DataSource.YFINANCE,
                            is_checked=1,
                            is_market_closed=0,
                        )
                        session.add(new_price)
                        records_added += 1

                        if records_added % 100 == 0:
                            session.commit()
                            logger.debug(
                                f"Промежуточный коммит: добавлено {records_added} записей"
                            )

            session.commit()
            logger.success(f"Добавлено {records_added} новых записей для {metal_name}")
            return records_added

        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка при обновлении данных для {metal_name}: {e}")
            raise
        finally:
            session.close()

    def start_scheduler(self):
        """Запускает планировщик задач в отдельном потоке"""
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            logger.warning("Планировщик уже запущен")
            return

        self._scheduler_thread = threading.Thread(target=self._run_task_scheduler)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
        logger.info("Планировщик задач запущен в отдельном потоке")

    def _run_task_scheduler(self):
        """Планировщик задач для обновления данных металлов"""
        logger.info("Запуск планировщика задач для обновления данных металлов")

        try:
            session = self.Session()
            try:
                active_schedules = (
                    session.query(CollectorSchedule)
                    .filter(CollectorSchedule.is_active == 1)
                    .all()
                )

                logger.info(
                    f"Найдено {len(active_schedules)} активных расписаний. Принудительно запускаем все задачи."
                )

                for schedule in active_schedules:
                    task_key = f"{schedule.metal_type.value}_{schedule.source.value}"

                    if (
                        task_key in self.running_tasks
                        and self.running_tasks[task_key]["running"]
                    ):
                        logger.debug(
                            f"Задача для {task_key} уже выполняется, пропускаем"
                        )
                        continue

                    now = datetime.now()
                    schedule.last_triggered = now
                    session.commit()

                    logger.info(
                        f"Начинаем задачу обновления данных для {schedule.metal_type.value} из источника {schedule.source.value}"
                    )

                    def task_runner(collector, metal_type, source, schedule_id):
                        try:
                            collector.running_tasks[
                                f"{metal_type.value}_{source.value}"
                            ] = {"running": True}

                            logger.info(
                                f"Выполнение обновления для {metal_type.value} из источника {source.value}"
                            )
                            records_added = collector.update_missing_metal_prices(
                                metal_type.value
                            )

                            with collector.Session() as update_session:
                                schedule = update_session.query(CollectorSchedule).get(
                                    schedule_id
                                )
                                if schedule:
                                    schedule.last_run = datetime.now()
                                    update_session.commit()
                                    logger.info(
                                        f"Обновлено время последнего запуска для задачи ID {schedule_id}"
                                    )

                            logger.success(
                                f"Задача для {metal_type.value} из источника {source.value} завершена успешно. Добавлено {records_added} записей"
                            )
                        except Exception as e:
                            logger.error(
                                f"Ошибка при выполнении задачи для {metal_type.value} из источника {source.value}: {e}"
                            )
                        finally:
                            collector.running_tasks[
                                f"{metal_type.value}_{source.value}"
                            ] = {"running": False}

                    task_thread = threading.Thread(
                        target=task_runner,
                        args=(self, schedule.metal_type, schedule.source, schedule.id),
                    )
                    task_thread.daemon = True
                    task_thread.start()

                    self.running_tasks[task_key] = {
                        "running": True,
                        "thread": task_thread,
                    }

            finally:
                session.close()

            logger.info("Ожидаем завершения принудительно запущенных задач...")
            time.sleep(5)

            while True:
                try:
                    session = self.Session()
                    now = datetime.now()

                    active_schedules = (
                        session.query(CollectorSchedule)
                        .filter(CollectorSchedule.is_active == 1)
                        .all()
                    )

                    schedule_info = []
                    for schedule in active_schedules:
                        next_run_time = None
                        seconds_until_next_run = "N/A"

                        if schedule.last_run:
                            interval_minutes = schedule.interval_minutes
                            next_run_time = schedule.last_run + timedelta(
                                minutes=interval_minutes
                            )

                            if next_run_time > now:
                                time_diff = next_run_time - now
                                seconds_until_next_run = (
                                    f"{int(time_diff.total_seconds())} сек."
                                )
                            else:
                                seconds_until_next_run = "0 сек. (просрочено)"

                        schedule_info.append(
                            f"{schedule.metal_type.value}+{schedule.source.value}: {seconds_until_next_run}"
                        )

                    logger.debug(
                        f"Найдено {len(active_schedules)} активных расписаний. Следующие запуски: {', '.join(schedule_info)}"
                    )

                    for schedule in active_schedules:
                        should_run = False

                        if not schedule.last_run:
                            should_run = True
                            logger.info(
                                f"Задача ID {schedule.id} ({schedule.metal_type.value}) никогда не запускалась"
                            )
                        else:
                            interval_minutes = schedule.interval_minutes
                            next_run_time = schedule.last_run + timedelta(
                                minutes=interval_minutes
                            )

                            if now >= next_run_time:
                                should_run = True
                                logger.info(
                                    f"Пора запустить задачу ID {schedule.id} ({schedule.metal_type.value}). "
                                    f"Последний запуск: {schedule.last_run}, "
                                    f"следующий запуск: {next_run_time}"
                                )

                        task_key = (
                            f"{schedule.metal_type.value}_{schedule.source.value}"
                        )
                        if (
                            task_key in self.running_tasks
                            and self.running_tasks[task_key]["running"]
                        ):
                            logger.debug(
                                f"Задача для {task_key} уже выполняется, пропускаем"
                            )
                            continue

                        if should_run:
                            schedule.last_triggered = now
                            session.commit()

                            logger.info(
                                f"Начинаем задачу обновления данных для {schedule.metal_type.value} из источника {schedule.source.value}"
                            )

                            def task_runner(collector, metal_type, source, schedule_id):
                                try:
                                    collector.running_tasks[
                                        f"{metal_type.value}_{source.value}"
                                    ] = {"running": True}

                                    logger.info(
                                        f"Выполнение обновления для {metal_type.value} из источника {source.value}"
                                    )
                                    records_added = (
                                        collector.update_missing_metal_prices(
                                            metal_type.value
                                        )
                                    )

                                    with collector.Session() as update_session:
                                        schedule = update_session.query(
                                            CollectorSchedule
                                        ).get(schedule_id)
                                        if schedule:
                                            schedule.last_run = datetime.now()
                                            update_session.commit()
                                            logger.info(
                                                f"Обновлено время последнего запуска для задачи ID {schedule_id}"
                                            )

                                    logger.success(
                                        f"Задача для {metal_type.value} из источника {source.value} завершена успешно. Добавлено {records_added} записей"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Ошибка при выполнении задачи для {metal_type.value} из источника {source.value}: {e}"
                                    )
                                finally:
                                    collector.running_tasks[
                                        f"{metal_type.value}_{source.value}"
                                    ] = {"running": False}

                            task_thread = threading.Thread(
                                target=task_runner,
                                args=(
                                    self,
                                    schedule.metal_type,
                                    schedule.source,
                                    schedule.id,
                                ),
                            )
                            task_thread.daemon = True
                            task_thread.start()

                            self.running_tasks[task_key] = {
                                "running": True,
                                "thread": task_thread,
                            }

                except Exception as e:
                    logger.error(f"Ошибка при проверке расписаний: {e}")
                finally:
                    session.close()

                time.sleep(5)

        except KeyboardInterrupt:
            logger.warning("Планировщик задач остановлен по запросу пользователя")

    def run_scheduled_tasks(self):
        """Запускает все активные задачи принудительно"""
        session = self.Session()
        results = {}

        try:
            now = datetime.now()

            active_schedules = (
                session.query(CollectorSchedule)
                .filter(CollectorSchedule.is_active == 1)
                .all()
            )

            logger.info(
                f"Принудительно запускаем {len(active_schedules)} активных расписаний"
            )

            for schedule in active_schedules:
                task_key = f"{schedule.metal_type.value}_{schedule.source.value}"

                if (
                    task_key in self.running_tasks
                    and self.running_tasks[task_key]["running"]
                ):
                    logger.debug(f"Задача для {task_key} уже выполняется, пропускаем")
                    results[task_key] = False
                    continue

                try:
                    schedule.last_triggered = now
                    session.commit()

                    self.running_tasks[task_key] = {"running": True}

                    logger.info(
                        f"Принудительное выполнение обновления для {schedule.metal_type.value} из источника {schedule.source.value}"
                    )
                    records_added = self.update_missing_metal_prices(
                        schedule.metal_type.value
                    )

                    schedule.last_run = datetime.now()
                    session.commit()

                    logger.success(
                        f"Задача для {schedule.metal_type.value} из источника {schedule.source.value} завершена. Добавлено {records_added} записей"
                    )
                    results[task_key] = True
                except Exception as e:
                    logger.error(
                        f"Ошибка при выполнении задачи для {schedule.metal_type.value} из источника {schedule.source.value}: {e}"
                    )
                    results[task_key] = False
                finally:
                    self.running_tasks[task_key] = {"running": False}

            return results

        except Exception as e:
            logger.error(f"Ошибка при выполнении запланированных задач: {e}")
            return {}
        finally:
            session.close()
