
from collections import defaultdict

data = open(r'C:\Users\Madina\PycharmProjects\pythonProject3\YOLOv8-SMOT\results\predictions\7s_frames\paraplane_conf02\tracks\7s_frames.txt').readlines()

tracks = defaultdict(list)
for line in data:
    parts = line.strip().split(',')
    frame_id = int(float(parts[0]))
    track_id = int(float(parts[1]))
    tracks[track_id].append(frame_id)

print(f'Всего уникальных ID: {len(tracks)}')
print()
print(f'{"ID":<6} {"Кадров":<10} {"Первый кадр":<15} {"Последний кадр":<15}')
print("-" * 46)
for tid in sorted(tracks.keys()):
    frames = tracks[tid]
    print(f'{tid:<6} {len(frames):<10} {min(frames):<15} {max(frames):<15}')

short = [tid for tid, frames in tracks.items() if len(frames) < 10]
long_ = [tid for tid, frames in tracks.items() if len(frames) >= 10]
print()
print(f'Короткие треки (< 10 кадров): {len(short)} штук — скорее всего мусор')
print(f'Длинные треки (>= 10 кадров): {len(long_)} штук — скорее всего реальный параплан')