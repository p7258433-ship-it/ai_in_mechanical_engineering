import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
import numpy as np
import time
import random
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt


class TensorFlowNeuralNetwork:

    def __init__(self, input_size: int = 3, hidden_size: int = 8, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = self._build_model()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='mse',
            metrics=['mae']
        )

        print(f"Нейронная сеть создана: {input_size} → {hidden_size} → {output_size}")

    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([

            tf.keras.layers.Input(shape=(self.input_size,)),

            tf.keras.layers.Dense(
                self.hidden_size,
                activation='relu',
                kernel_initializer='he_normal',
                name='hidden_layer'
            ),

            tf.keras.layers.Dense(
                self.hidden_size // 2,
                activation='relu',
                name='hidden_layer_2'
            ),

            tf.keras.layers.Dense(
                self.output_size,
                activation=None,
                name='output_layer'
            )
        ])

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 200, batch_size: int = 32,
              validation_split: float = 0.2) -> tf.keras.callbacks.History:
        print(f"Начало обучения: {epochs} эпох, batch_size={batch_size}")

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            callbacks=[

                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )

        print("Обучение завершено")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

        predictions = self.model.predict(X_tensor, verbose=0)

        return predictions

    def save_model(self, filepath: str = 'engine_air_mass_model.h5'):
        self.model.save(filepath)
        print(f"Модель сохранена в {filepath}")

    def load_model(self, filepath: str = 'engine_air_mass_model.h5'):
        self.model = tf.keras.models.load_model(filepath)
        print(f"Модель загружена из {filepath}")


class EngineControlSystemAI:

    def __init__(self, air_mass_threshold: float = 0.25):

        self.air_mass_threshold = air_mass_threshold

        self.nn = TensorFlowNeuralNetwork(input_size=3, hidden_size=16, output_size=1)

        self.sensor_faults = {
            'signal_hang': False,
            'maf_sensor_fault': False,
            'map_sensor_fault': False,
            'temperature_fault': False
        }

        self.operation_stats = {
            'total_cycles': 0,
            'normal_operations': 0,
            'ai_corrections': 0,
            'sensor_faults_detected': 0,
            'check_engine_signals': 0,
            'successful_ai_corrections': 0,
            'failed_ai_corrections': 0
        }

        self.training_data = []
        self.training_labels = []
        self.is_trained = False

        self.NORMALIZATION = {
            'rpm': {'max': 3000, 'min': 800},
            'tps': {'max': 100, 'min': 0},
            'temperature': {'max': 120, 'min': 80},
            'air_mass': {'max': 250, 'min': 10}
        }

        self.successful_ai_corrections = 0
        self.failed_ai_corrections = 0

        print("Система управления двигателем с ИИ инициализирована")

    def normalize_data(self, rpm: float, tps: float, temperature: float) -> np.ndarray:

        rpm_norm = (rpm - self.NORMALIZATION['rpm']['min']) / \
                   (self.NORMALIZATION['rpm']['max'] - self.NORMALIZATION['rpm']['min'])

        tps_norm = (tps - self.NORMALIZATION['tps']['min']) / \
                   (self.NORMALIZATION['tps']['max'] - self.NORMALIZATION['tps']['min'])

        temp_norm = (temperature - self.NORMALIZATION['temperature']['min']) / \
                    (self.NORMALIZATION['temperature']['max'] - self.NORMALIZATION['temperature']['min'])

        rpm_norm = np.clip(rpm_norm, 0, 1)
        tps_norm = np.clip(tps_norm, 0, 1)
        temp_norm = np.clip(temp_norm, 0, 1)

        return np.array([[rpm_norm, tps_norm, temp_norm]])

    def denormalize_air_mass(self, normalized_air_mass: float) -> float:

        min_p = self.NORMALIZATION['air_mass']['min']
        max_p = self.NORMALIZATION['air_mass']['max']

        real_air_mass = min_p + normalized_air_mass * (max_p - min_p)

        return real_air_mass

    def collect_sensor_data(self) -> Tuple[float, float, float]:

        rpm = random.uniform(800, 3000)
        tps = random.uniform(0, 100)
        temperature = random.uniform(80, 110)

        return rpm, tps, temperature

    def calculate_expected_air_mass(self, rpm: float, tps: float, temperature: float) -> float:

        base_air_mass = 30.0  # кПа

        rpm_factor = rpm / 2000

        tps_factor = 0.3 + 0.7 * (tps / 100) ** 1.5

        temp_factor = 1.2 - 0.003 * (temperature - 90)

        expected_air_mass = base_air_mass * rpm_factor * tps_factor * temp_factor

        air_mass_noise = random.uniform(-3, 3)
        expected_air_mass += air_mass_noise

        expected_air_mass = max(10, min(250, expected_air_mass))

        return expected_air_mass

    def read_maf_sensor(self, rpm: float, tps: float, temperature: float) -> Optional[float]:

        fault_probability = 0.12

        rand_val = random.random()

        normal_air_mass = self.calculate_expected_air_mass(rpm, tps, temperature)

        sensor_noise = random.uniform(-0.05, 0.05)
        sensor_air_mass = normal_air_mass * (1 + sensor_noise)

        sensor_air_mass += random.uniform(-2, 2)

        sensor_air_mass = max(10, min(250, sensor_air_mass))

        if rand_val < 0.05:
            self.sensor_faults['signal_hang'] = True
            print("Зависание сигнала ДМРВ")
            return None

        elif rand_val < 0.09:
            self.sensor_faults['maf_sensor_fault'] = True
            print("ДМРВ неисправен")
            return None

        elif rand_val < 0.12:
            self.sensor_faults['temperature_fault'] = True
            print("Датчик температуры воздуха неисправен")
            return None

        return sensor_air_mass

    def calculate_air_mass_deviation(self, expected: float, actual: Optional[float]) -> Optional[float]:

        if actual is None:
            return None

        if random.random() < 0.3:
            actual = expected * random.uniform(0.99, 1.01)

        deviation = abs(actual - expected) / expected

        return deviation

    def train_ai_system(self, num_samples: int = 500):
        #
        # print("\n" + "=" * 60)
        # print("ОБУЧЕНИЕ АЛГОРИТМУ РАБОТЫ БАЗОВОГО ДАТЧИКА")
        # print("=" * 60)

        X_train = []
        y_train = []

        for i in range(num_samples):

            rpm = random.uniform(800, 3000)
            tps = random.uniform(0, 100)
            temperature = random.uniform(80, 110)

            expected_air_mass = self.calculate_expected_air_mass(rpm, tps, temperature)

            X_normalized = self.normalize_data(rpm, tps, temperature)[0]

            air_mass_normalized = (expected_air_mass - self.NORMALIZATION['air_mass']['min']) / \
                                  (self.NORMALIZATION['air_mass']['max'] - self.NORMALIZATION['air_mass']['min'])

            X_train.append(X_normalized)
            y_train.append([air_mass_normalized])

            # if i % 100 == 0:
            #     print(f"  Генерация данных: {i}/{num_samples}")

        X_train_np = np.array(X_train, dtype=np.float32)
        y_train_np = np.array(y_train, dtype=np.float32)

        print(f"\nСгенерировано {len(X_train)} примеров для обучения")
        print(f"   Входные данные: {X_train_np.shape}")
        print(f"   Целевые значения: {y_train_np.shape}")

        print("\nНачало обучения нейронной сети...")
        history = self.nn.train(
            X_train_np,
            y_train_np,
            epochs=150,
            batch_size=32,
            validation_split=0.2
        )

        self.training_data = X_train_np
        self.training_labels = y_train_np
        self.is_trained = True

        self._plot_training_history(history)

        self.nn.save_model('maf_sensor_model.h5')

    def _plot_training_history(self, history: tf.keras.callbacks.History):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history['loss'], label='Ошибка обучения')
        ax1.plot(history.history['val_loss'], label='Ошибка валидации')
        ax1.set_title('Функция потерь (MSE)')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Ошибка')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history['mae'], label='MAE обучения')
        ax2.plot(history.history['val_mae'], label='MAE валидации')
        ax2.set_title('Средняя абсолютная ошибка (MAE)')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100)
        print("Графики обучения сохранены в 'training_history.png'")

    def ai_correct_sensor_data(self, rpm: float, tps: float, temperature: float) -> float:

        if not self.is_trained:
            raise RuntimeError("Нейронная сеть не обучена!")

        X_normalized = self.normalize_data(rpm, tps, temperature)
        predicted_normalized = self.nn.predict(X_normalized)[0][0]
        corrected_air_mass = self.denormalize_air_mass(predicted_normalized)

        expected_air_mass = self.calculate_expected_air_mass(rpm, tps, temperature)
        correction_error = abs(corrected_air_mass - expected_air_mass) / expected_air_mass

        if correction_error< 0.15:
            self.successful_ai_corrections += 1
        else:
            self.failed_ai_corrections += 1

        self.operation_stats['ai_corrections'] += 1

        return corrected_air_mass

    def check_engine_system(self) -> Dict[str, bool]:

        system_status = {
            'check_engine': False,
            'system_fault': False
        }

        if self.sensor_faults.get('signal_hang', False):
            print("Сигнал 'Check Engine': обнаружено зависание сигнала ДМРВ")
            system_status['check_engine'] = True
            system_status['system_fault'] = True
            self.operation_stats['check_engine_signals'] += 1

        if self.sensor_faults.get('maf_sensor_fault', False):
            print("Сигнал о неисправности системы: ДМРВ требует замены")
            system_status['system_fault'] = True

        if self.sensor_faults.get('temperature_fault', False):
            print("Сигнал о неисправности системы: проверьте датчик температуры воздуха")
            system_status['system_fault'] = True

        for fault in self.sensor_faults:
            self.sensor_faults[fault] = False

        return system_status

    def engine_control_cycle(self) -> bool:

        self.operation_stats['total_cycles'] += 1

        print("\n")
        print(f"ЦИКЛ РАБОТЫ #{self.operation_stats['total_cycles']}")

        print("1. Сбор данных с датчиков...")
        rpm, tps, temperature = self.collect_sensor_data()
        print(f"     Обороты: {rpm:.0f} об/мин")
        print(f"     Дроссель: {tps:.1f}%")
        print(f"     Температура: {temperature:.1f}°C")

        print("2. Расчет ожидаемого расхода воздуха...")
        expected_air_mass = self.calculate_expected_air_mass(rpm, tps, temperature)
        print(f"     Ожидаемый расход воздуха: {expected_air_mass:.2f} г/с")

        print("3. Чтение датчика массового расхода воздуха (ДМРВ)...")
        actual_air_mass = self.read_maf_sensor(rpm, tps, temperature)

        if actual_air_mass is not None:
            print(f"     Фактическое расход: {actual_air_mass:.2f} г/с")

            difference = actual_air_mass - expected_air_mass
            print(f"     Разница с расчетами: {difference:+.2f} г/с")


        print("4. Расчет отклонений...")
        air_mass_deviation = self.calculate_air_mass_deviation(expected_air_mass, actual_air_mass)

        if air_mass_deviation is not None:
            deviation_percent = air_mass_deviation * 100
            print(f"     Отклонение: {deviation_percent:.1f}%")

            if deviation_percent< 5:
                print("    Данные ДМРВ совпадают с расчетными")

            if deviation_percent > self.air_mass_threshold * 100:
                print(f"    Отклонение превышает порог ({self.air_mass_threshold * 100:.0f}%)")
                self.operation_stats['sensor_faults_detected'] += 1

        print("5. Диагностика системы...")
        system_status = self.check_engine_system()

        needs_correction = (
                actual_air_mass is None or
                (air_mass_deviation is not None and air_mass_deviation > self.air_mass_threshold) or
                system_status['system_fault']
        )

        if needs_correction:
            if self.is_trained:
                print("6. Активация ИИ-коррекции...")
                corrected_air_mass = self.ai_correct_sensor_data(rpm, tps, temperature)
                print(f"• Расход воздуха от ИИ: {corrected_air_mass:.2f} г/с")
                print("  Продолжение работы с ИИ-регулированием")
                return True
            else:
                print("6. Система не обучена, требуется ручное вмешательство!")
                return False
        else:
            self.operation_stats['normal_operations'] += 1
            print("6. Все системы в норме, продолжение работы")
            return True

    def run_system_simulation(self, num_cycles: int = 25):

        # print("\n" + "=" * 60)
        print("ЗАПУСК СИМУЛЯЦИИ СИСТЕМЫ УПРАВЛЕНИЯ ДВИГАТЕЛЕМ")
        # print("=" * 60)

        print("\n Этап 1: Подготовка и обучение системы...")
        self.train_ai_system(num_samples=400)

        print("\n Этап 2: Запуск рабочих циклов...")
        successful_cycles = 0

        for cycle in range(num_cycles):
            success = self.engine_control_cycle()

            if success:
                successful_cycles += 1

            time.sleep(0.3)

        self._print_final_statistics(num_cycles, successful_cycles)

    # def _print_final_statistics(self, total_cycles: int, successful_cycles: int):
    #
    #     print("\n" + "=" * 60)
    #     print("ИТОГОВАЯ СТАТИСТИКА РАБОТЫ СИСТЕМЫ")
    #     print("=" * 60)
    #
    #     stats = self.operation_stats
    #
    #     print(f"\nОбщая статистика:")
    #     print(f" • Всего циклов работы: {stats['total_cycles']}")
    #     print(f" • Успешных циклов: {successful_cycles}")
    #     print(f" • Эффективность системы: {successful_cycles / total_cycles * 100:.1f}%")
    #
    #     print(f"\nСтатистика работы датчиков:")
    #     print(f" • Нормальных операций: {stats['normal_operations']}")
    #     print(f" • Коррекций ИИ: {stats['ai_corrections']}")
    #     print(f" • Обнаружено неисправностей: {stats['sensor_faults_detected']}")
    #     print(f" • Сигналов 'Check Engine': {stats['check_engine_signals']}")
    #
    #     print(f"\nЭффективность ИИ:")
    #     if stats['sensor_faults_detected'] > 0:
    #         ai_success_rate = (stats['ai_corrections'] /
    #                            (stats['sensor_faults_detected'] + stats['check_engine_signals'])) * 100
    #         print(f"   • Успешных коррекций: {ai_success_rate:.1f}% от неисправностей")
    #
    #     print(f"\nРекомендации:")
    #     if stats['sensor_faults_detected'] / total_cycles > 0.1:
    #         print("   Высокий уровень неисправностей. Рекомендуется диагностика.")
    #     else:
    #         print("   Система работает стабильно.")
    #
    #     if stats['ai_corrections'] > 0:
    #         print(f"   ИИ предотвратил {stats['ai_corrections']} остановок двигателя")


    def _print_final_statistics(self, total_cycles: int, successful_cycles: int):

        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА РАБОТЫ СИСТЕМЫ")
        print("=" * 60)

        stats = self.operation_stats

        print(f"\nОбщая статистика:")
        print(f" • Всего циклов работы: {stats['total_cycles']}")
        print(f" • Успешных циклов: {successful_cycles}")
        print(f" • Эффективность системы: {successful_cycles / total_cycles * 100:.1f}%")

        print(f"\nСтатистика работы датчиков:")
        print(f" • Нормальных операций: {stats['normal_operations']}")
        print(f" • Всего коррекций ИИ: {stats['ai_corrections']}")
        print(f" • Обнаружено неисправностей: {stats['sensor_faults_detected']}")
        print(f" • Сигналов 'Check Engine': {stats['check_engine_signals']}")

        print(f"\nЭффективность ИИ:")

        total_faults = stats['sensor_faults_detected'] + stats['check_engine_signals']

        if total_faults > 0:
            ai_coverage_rate = (stats['ai_corrections'] / total_faults) * 100
            print(f"   • ИИ обработал {ai_coverage_rate:.1f}% неисправностей")

            if hasattr(self, 'successful_ai_corrections'):
                if stats['ai_corrections'] > 0:
                    ai_success_rate = (self.successful_ai_corrections /
                                       stats['ai_corrections']) * 100
                    print(f"   • Успешных коррекций: {ai_success_rate:.1f}%")
        else:
            print("   • Неисправностей не обнаружено")

        if stats['ai_corrections'] > 0 and total_cycles > 0:
            ai_usage_rate = (stats['ai_corrections'] / total_cycles) * 100
            print(f"   • Частота использования ИИ: {ai_usage_rate:.1f}% циклов")

        print(f"\nРекомендации:")
        if total_cycles > 0:
            fault_rate = total_faults / total_cycles * 100
            if fault_rate > 15:
                print(f"       Высокий уровень неисправностей ({fault_rate:.1f}%). Требуется диагностика.")
            elif fault_rate > 5:
                print(f"       Умеренный уровень неисправностей ({fault_rate:.1f}%). Рекомендуется проверка.")
            else:
                print(f"     Система работает стабильно (неисправности: {fault_rate:.1f}%).")

        if stats['ai_corrections'] > 0:
            print(f"     ИИ предотвратил {stats['ai_corrections']} потенциальных остановок двигателя")

        if successful_cycles == total_cycles:
            print("     Все циклы завершены успешно!")

def main():

    engine_system = EngineControlSystemAI(air_mass_threshold=0.15)

    engine_system.run_system_simulation(num_cycles=25)

    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 60)

    test_rpm = 2200
    test_tps = 65
    test_temp = 95

    print(f"\nТестовые данные двигателя:")
    print(f"  • Обороты: {test_rpm} об/мин")
    print(f"  • Положение дросселя: {test_tps}%")
    print(f"  • Температура: {test_temp}°C")

    expected = engine_system.calculate_expected_air_mass(test_rpm, test_tps, test_temp)
    print(f"\nОжидаемый расход воздуха (физическая модель): {expected:.2f} г/с")

    if engine_system.is_trained:
        predicted = engine_system.ai_correct_sensor_data(test_rpm, test_tps, test_temp)
        print(f"Прогноз нейронной сети: {predicted:.2f} г/с")

        deviation = abs(predicted - expected) / expected * 100
        print(f"Отклонение прогноза: {deviation:.2f}%")

        if deviation < 5:
            print("Точность прогноза: отличная")
        elif deviation < 10:
            print("Точность прогноза: удовлетворительная")
        else:
            print("Точность прогноза: требует улучшения")


if __name__ == "__main__":
    main()

    print("\n")
    print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
    # print("=" * 60)
    print("\nВыполнены все этапы алгоритма:")
    print("1. Обучение алгоритму работы базового датчика")
    print("2. Сбор начальных данных с датчиков")
    print("3. Расчет ожидаемого давления")
    print("4. Чтение сигнальных давлений с ДАД")
    print("5. Расчет отклонений")
    print("6. Обнаружение зависания сигнала")
    print("7. Диагностика неисправностей датчиков")
    print("8. Сигнализация 'Check Engine' при неисправностях")
    print("9. Проверка превышения порога отклонений")
    print("10. Замена некорректных данных с помощью ИИ")
    print("11. Продолжение работы двигателя")
