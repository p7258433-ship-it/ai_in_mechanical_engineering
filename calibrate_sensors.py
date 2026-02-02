#!/usr/bin/env python3
"""
Калибровка датчиков перед использованием
"""
import time
from sensor_interface import RealSensorInterface


def calibrate_tps():
    """Калибровка датчика положения дросселя"""
    print("\n=== Калибровка TPS ===")
    print("1. Полностью закройте дроссель (0%)")
    input("Нажмите Enter когда готово...")

    sensor = RealSensorInterface(simulation_mode=False)
    voltage_min = sensor.read_maf_voltage()  # Используем метод чтения напряжения

    print(f"Минимальное напряжение: {voltage_min:.2f}V")

    print("\n2. Полностью откройте дроссель (100%)")
    input("Нажмите Enter когда готово...")

    voltage_max = sensor.read_maf_voltage()
    print(f"Максимальное напряжение: {voltage_max:.2f}V")

    sensor.cleanup()
    return voltage_min, voltage_max


if name == "main":
    print("Программа калибровки датчиков")
    print("=" * 50)

    # Калибровка TPS
    v_min, v_max = calibrate_tps()

    print(f"\nРекомендуемые настройки для config.json:")
    print(f'"voltage_min": {v_min:.2f},')
    print(f'"voltage_max": {v_max:.2f}')

    # Сохранение калибровки
    with open('manual_calibration.txt', 'w') as f:
        f.write(f"TPS калибровка:\n")
        f.write(f"  Min voltage: {v_min:.2f}V\n")
        f.write(f"  Max voltage: {v_max:.2f}V\n")

    print("\nКалибровка завершена!")