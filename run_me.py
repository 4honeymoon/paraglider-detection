import os
import sys
import subprocess

# 1. Принудительно прописываем путь к текущей папке в систему
current_dir = os.getcwd()
os.environ['PYTHONPATH'] = current_dir + os.pathsep + os.environ.get('PYTHONPATH', '')

# 2. Проверяем, видит ли Python папку mmyolo
try:
    import mmyolo
    print("--- УРА! mmyolo найден ---")
except ImportError:
    print("--- КРИТИЧЕСКАЯ ОШИБКА: Папка mmyolo не найдена в", current_dir)
    sys.exit()

# 3. Запускаем основной скрипт через системный вызов с правильными путями
cmd = [
    "python", "tools/predict.py",
    "--path", "test.mp4",
    "--save_result",
    "--device", "cpu"
]

print("Запускаю предсказание...")
subprocess.run(cmd, env=os.environ)