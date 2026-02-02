"""
Интерфейс для работы с реальными датчиками на Raspberry Pi
Конкретно для:
- Аналоговый ДМРВ → MCP3008 Channel 0
- Потенциометр WH148-B1K → MCP3008 Channel 1
- DS18B20 → GPIO4 + 1-Wire
- Датчик Холла Troyka → GPIO17
"""
import time
import json
import os
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

# GPIO библиотека
try:
    import RPi.GPIO as GPIO

    RASPBERRY_AVAILABLE = True
    print("Библиотека RPi.GPIO загружена успешно")
except ImportError:
    RASPBERRY_AVAILABLE = False
    print("Режим эмуляции: RPi.GPIO не найден")

# MCP3008 (АЦП) библиотека
try:
    import busio
    import digitalio
    import board
    import adafruit_mcp3xxx.mcp3008 as MCP
    from adafruit_mcp3xxx.analog_in import AnalogIn

    MCP3008_AVAILABLE = True
    print("Библиотека MCP3008 загружена успешно")
except ImportError:
    MCP3008_AVAILABLE = False
    print("Режим эмуляции: MCP3008 библиотеки не найдены")

# Для DS18B20 (1-Wire интерфейс)
DS18B20_AVAILABLE = os.path.exists('/sys/bus/w1/devices/')


@dataclass
class SensorCalibration:
    """Калибровочные данные для датчиков"""
    maf_voltage_min: float = 0.5  # Вольт
    maf_voltage_max: float = 4.5  # Вольт
    maf_flow_min: float = 0.0  # г/с
    maf_flow_max: float = 300.0  # г/с
    tps_voltage_min: float = 0.5  # Вольт
    tps_voltage_max: float = 4.5  # Вольт
    hall_pulses_per_rev: int = 2  # Импульсов на оборот
    temp_correction: float = 0.0  # Коррекция температуры


class RealSensorInterface:
    """Интерфейс для работы с реальными датчиками"""

    def init(self, simulation_mode: bool = False,
             data_log_file: str = "sensor_data_log.csv"):
        self.simulation_mode = simulation_mode
        self.data_log_file = data_log_file
        self.calibration = SensorCalibration()

        # Статистика и логирование
        self.pulse_count = 0
        self.last_pulse_time = time.time()
        self.last_rpm_calc = 0
        self.log_buffer = []

        # Загрузка калибровки
        self._load_calibration()

        if not simulation_mode:
            self._setup_sensors()
            self._setup_data_logging()
        else:
            print("Режим работы: симуляция датчиков")

    def _load_calibration(self):
        """Загрузка калибровочных данных из файла"""
        try:
            if os.path.exists('sensor_calibration.json'):
                with open('sensor_calibration.json', 'r') as f:
                    calib_data = json.load(f)
                    self.calibration = SensorCalibration(**calib_data)
                    print("Калибровочные данные загружены")
        except Exception as e:
            print(f"Ошибка загрузки калибровки: {e}, используются значения по умолчанию")

    def _save_calibration(self):
        """Сохранение калибровочных данных"""
        try:
            with open('sensor_calibration.json', 'w') as f:
                json.dump(self.calibration.dict, f, indent=4)
                print("Калибровочные данные сохранены")
        except Exception as e:
            print(f"Ошибка сохранения калибровки: {e}")

    def _setup_data_logging(self):
        """Настройка логирования данных"""
        try:
            # Создаем заголовок CSV файла если файл не существует
            if not os.path.exists(self.data_log_file):
                with open(self.data_log_file, 'w') as f:
                    f.write("timestamp,rpm,tps,temperature,maf_voltage,maf_flow,ai_correction\n")
                print(f"Файл логов создан: {self.data_log_file}")
        except Exception as e:
            print(f"Ошибка настройки логирования: {e}")

    def _log_sensor_data(self, rpm: float, tps: float, temp: float,
                         maf_voltage: float, maf_flow: float, ai_correction: bool = False):
        """Логирование данных с датчиков"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_line = f"{timestamp},{rpm:.1f},{tps:.1f},{temp:.1f},{maf_voltage:.3f},{maf_flow:.1f},{int(ai_correction)}\n"

            self.log_buffer.append(log_line)

            # Записываем в файл каждые 10 записей
            if len(self.log_buffer) >= 10:
                with open(self.data_log_file, 'a') as f:
                    f.writelines(self.log_buffer)
                self.log_buffer = []

        except Exception as e:
            print(f"Ошибка логирования: {e}")

    def _setup_sensors(self):
        """Настройка всех датчиков"""
        if self.simulation_mode:
            return

        # Настройка GPIO
        if RASPBERRY_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Датчик Холла на GPIO17
            GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # Добавляем прерывание для подсчета импульсов
            GPIO.add_event_detect(17, GPIO.FALLING,
                                  callback=self._hall_sensor_callback,
                                  bouncetime=10)
            print("Датчик Холла настроен на GPIO17")

        # Настройка MCP3008 (АЦП)
        if MCP3008_AVAILABLE:
            try:
                # Создаем SPI шину
                spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

                # Создаем CS (chip select)
                cs = digitalio.DigitalInOut(board.D8)  # CE0 на GPIO8

                # Создаем объект MCP3008
                self.mcp = MCP.MCP3008(spi, cs)

                # Создаем аналоговые входы
                self.maf_channel = AnalogIn(self.mcp, MCP.P0)  # ДМРВ на Channel 0
                self.tps_channel = AnalogIn(self.mcp, MCP.P1)  # Потенциометр на Channel 1

                print(f"MCP3008 настроен. ДМРВ на P0, TPS на P1")
                print(f"Напряжение опорное: {self.maf_channel.reference_voltage}V")

            except Exception as e:
                print(f"Ошибка настройки MCP3008: {e}")
                MCP3008_AVAILABLE = False

        # Настройка DS18B20 (1-Wire)
        if DS18B20_AVAILABLE:
            try:
                # Включаем 1-Wire интерфейс
                os.system('sudo modprobe w1-gpio')
                os.system('sudo modprobe w1-therm')

                # Ищем устройство
                base_dir = '/sys/bus/w1/devices/'
                device_folders = [f for f in os.listdir(base_dir) if f.startswith('28-')]

                if device_folders:
                    self.ds18b20_device = os.path.join(base_dir, device_folders[0], 'w1_slave')
                    print(f"DS18B20 найден: {device_folders[0]}")
                else:
                    print("DS18B20 не найден")
                    DS18B20_AVAILABLE = False

            except Exception as e:
                print(f"Ошибка настройки DS18B20: {e}")
                DS18B20_AVAILABLE = False

        print("Настройка датчиков завершена")

    def _hall_sensor_callback(self, channel):
        """Обработчик прерывания датчика Холла"""
        current_time = time.time()

        # Фильтрация дребезга
        if current_time - self.last_pulse_time > 0.01:  # 10ms дебаунс
            self.pulse_count += 1
            self.last_pulse_time = current_time

    def read_rpm(self) -> float:
        """Чтение оборотов двигателя с датчика Холла"""
        if self.simulation_mode or not RASPBERRY_AVAILABLE:
            return 1500.0 + (time.time() % 10) * 100  # Плавное изменение для теста

            # Метод подсчета импульсов за интервал времени
            sample_time = 0.5  # 500ms для лучшей точности на низких оборотах

            # Сохраняем начальные значения
            start_count = self.pulse_count
            start_time = time.time()

            # Ждем sample_time секунд
            time.sleep(sample_time)

            # Вычисляем разницу
            end_count = self.pulse_count
            end_time = time.time()

            elapsed_time = end_time - start_time
            pulses = end_count - start_count

            if elapsed_time > 0 and pulses > 0:
                # RPM = (импульсы/секунду) * 60 / импульсов_на_оборот
                pulses_per_second = pulses / elapsed_time
                rpm = (pulses_per_second * 60) / self.calibration.hall_pulses_per_rev
                self.last_rpm_calc = max(0, min(8000, rpm))  # Ограничение 0-8000 RPM
            else:
                rpm = self.last_rpm_calc * 0.9  # Плавное снижение если нет импульсов

            return rpm

        def read_tps(self) -> float:
            """Чтение положения дроссельной заслонки с потенциометра"""
            if self.simulation_mode or not MCP3008_AVAILABLE:
                return 50.0 + (time.time() % 10) * 5  # Плавное изменение

            try:
                # Чтение напряжения с потенциометра
                voltage = self.tps_channel.voltage

                # Ограничение по диапазону
                voltage = max(self.calibration.tps_voltage_min,
                              min(self.calibration.tps_voltage_max, voltage))

                # Конвертация в проценты (0-100%)
                voltage_range = self.calibration.tps_voltage_max - self.calibration.tps_voltage_min
                if voltage_range > 0:
                    percentage = ((voltage - self.calibration.tps_voltage_min) / voltage_range) * 100
                else:
                    percentage = 50.0

                return max(0, min(100, percentage))

            except Exception as e:
                print(f"Ошибка чтения TPS: {e}")
                return 0.0

        def read_ds18b20_raw(self) -> str:
            """Чтение сырых данных с DS18B20"""
            try:
                with open(self.ds18b20_device, 'r') as f:
                    lines = f.readlines()
                return lines
            except:
                return None

        def read_temperature(self) -> float:
            """Чтение температуры с DS18B20"""
            if self.simulation_mode or not DS18B20_AVAILABLE:
                return 90.0 + (time.time() % 20)  # Плавное изменение

            try:
                lines = self.read_ds18b20_raw()
                if lines is None:
                    return 90.0

                # Проверка CRC
                while lines[0].strip()[-3:] != 'YES':
                    time.sleep(0.2)
                    lines = self.read_ds18b20_raw()
                    if lines is None:
                        return 90.0

                # Поиск температуры
                temp_pos = lines[1].find('t=')
                if temp_pos != -1:
                    temp_string = lines[1][temp_pos + 2:]
                    temp_c = float(temp_string) / 1000.0
                    return temp_c + self.calibration.temp_correction

            except Exception as e:
                print(f"Ошибка чтения температуры: {e}")

            return 90.0  # Значение по умолчанию

        def read_maf_voltage(self) -> float:
            """Чтение напряжения с ДМРВ"""
            if self.simulation_mode or not MCP3008_AVAILABLE:
                return 2.0 + (time.time() % 5) * 0.5  # Плавное изменение

            try:
                # Чтение напряжения с ДМРВ
                voltage = self.maf_channel.voltage

                # Ограничение по диапазону
                voltage = max(self.calibration.maf_voltage_min,
                              min(self.calibration.maf_voltage_max, voltage))

            return voltage

            except Exception as e:
            print(f"Ошибка чтения ДМРВ: {e}")
            return 0.0

    def read_maf(self) -> Optional[float]:
        """Чтение расхода воздуха с ДМРВ (г/с)"""
        voltage = self.read_maf_voltage()

        # Проверка на обрыв/короткое замыкание
        if voltage < self.calibration.maf_voltage_min * 0.8:
            print(f"Низкое напряжение ДМРВ: {voltage:.2f}V (возможно обрыв)")
            return None
        elif voltage > self.calibration.maf_voltage_max * 1.1:
            print(f"Высокое напряжение ДМРВ: {voltage:.2f}V (возможно КЗ)")
            return None

        # Линейная конвертация напряжения в расход
        voltage_range = self.calibration.maf_voltage_max - self.calibration.maf_voltage_min
        flow_range = self.calibration.maf_flow_max - self.calibration.maf_flow_min

        if voltage_range > 0:
            flow = self.calibration.maf_flow_min + \
                   ((voltage - self.calibration.maf_voltage_min) / voltage_range) * flow_range
        else:
            flow = self.calibration.maf_flow_min

        return max(0, flow)

    def calibrate_maf_sensor(self, known_flow: float):
        """Калибровка ДМРВ по известному расходу"""
        voltage = self.read_maf_voltage()

        if 0.5 < voltage < 4.5:  # Разумные пределы
            print(f"Калибровка ДМРВ: напряжение={voltage:.2f}V, расход={known_flow:.1f}г/с")

            # Можно обновить калибровку (упрощенный метод)
            self.calibration.maf_voltage_min = min(self.calibration.maf_voltage_min, voltage * 0.8)
            self.calibration.maf_voltage_max = max(self.calibration.maf_voltage_max, voltage * 1.2)

            if known_flow > self.calibration.maf_flow_max:
                self.calibration.maf_flow_max = known_flow * 1.1

            self._save_calibration()
            return True

        return False

    def read_all_sensors(self) -> Dict[str, float]:
        """Чтение всех датчиков одновременно"""
        rpm = self.read_rpm()
        tps = self.read_tps()
        temp = self.read_temperature()
        maf_voltage = self.read_maf_voltage()
        maf_flow = self.read_maf()

        # Логирование
        if not self.simulation_mode:
            self._log_sensor_data(rpm, tps, temp, maf_voltage,
                                  maf_flow if maf_flow else 0.0, False)

        return {
            'rpm': rpm,
            'tps': tps,
            'temperature': temp,
            'maf_voltage': maf_voltage,
            'maf_flow': maf_flow
        }

    def cleanup(self):
        """Очистка ресурсов"""
        if RASPBERRY_AVAILABLE:
            GPIO.cleanup()

        # Запись оставшихся логов
        if self.log_buffer:
            try:
                with open(self.data_log_file, 'a') as f:
                    f.writelines(self.log_buffer)
                self.log_buffer = []
            except:
                pass

        print("Ресурсы датчиков освобождены")

    # Простая тестовая программа
    if name == "main":
        print("Тест реальных датчиков")
        print("=" * 50)

        # Пробуем реальные датчики, если не получится - эмуляция
        try:
            sensor = RealSensorInterface(simulation_mode=False)
            print("Режим: РЕАЛЬНЫЕ ДАТЧИКИ")
        except:
            sensor = RealSensorInterface(simulation_mode=True)
            print("Режим: ЭМУЛЯЦИЯ")

        try:
            for i in range(10):
                data = sensor.read_all_sensors()
                print(f"\nЦикл {i + 1}:")
                print(f"  RPM: {data['rpm']:.0f} об/мин")
                print(f"  TPS: {data['tps']:.1f}%")
                print(f"  Температура: {data['temperature']:.1f}°C")
        print(f"  ДМРВ напряжение: {data['maf_voltage']:.2f}V")
        print(f"  ДМРВ расход: {data['maf_flow'] if data['maf_flow'] else 'Ошибка'} г/с")
        time.sleep(1)
    except KeyboardInterrupt:
    print("\nТест остановлен")

finally:
sensor.cleanup()
print("Тест завершен")