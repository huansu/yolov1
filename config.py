import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# --------------------------检测类别名--------------------------------------
class_names = (
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=416),
        A.PadIfNeeded(
            min_height= 416,
            min_width= 416,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),

        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1, 1, 1), max_pixel_value=255,),
        ToTensorV2(),
    ],
   #bbox_params=A.BboxParams(format="albumentations", min_visibility=0.4, label_fields=[],),
)