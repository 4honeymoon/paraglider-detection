custom_imports = dict(imports=['mmyolo'], allow_failed_imports=False)
default_scope = 'mmyolo'

_base_ = './yolov8_l.py'

# ======= Твой класс =======
metainfo = dict(classes=('paraplane',))
num_classes = 1

# ======= Путь к датасету =======
# Вставь сюда реальный путь к папке датасета на рабочем столе
data_root = 'C:/Users/Madina/Desktop/par/'

# ======= Модель =======
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)
    )
)

# ======= Размер изображения для обучения =======
# Нарезаем на патчи 640x640 т.к. объекты маленькие
imgsz = 640

# ======= Данные для обучения =======
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img='images/train/'),
    )
)

# ======= Данные для валидации =======
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img='images/val/'),
    )
)

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'data.yaml',
    metric='bbox'
)

# ======= Предобученные веса =======
load_from = 'C:/Users/Madina/PycharmProjects/pythonProject3/YOLOv8-SMOT/checkpoints/yolo8_l.pth'

# ======= Параметры обучения =======
train_cfg = dict(max_epochs=100, val_interval=5)

optim_wrapper = dict(
    optimizer=dict(lr=0.001)
)