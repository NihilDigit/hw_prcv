"""CamVid 数据集定义与加载器"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np

# 完整 32 类定义 (RGB -> 名称)
CAMVID_FULL_CLASSES = {
    (64, 128, 64): "Animal",
    (192, 0, 128): "Archway",
    (0, 128, 192): "Bicyclist",
    (0, 128, 64): "Bridge",
    (128, 0, 0): "Building",
    (64, 0, 128): "Car",
    (64, 0, 192): "CartLuggagePram",
    (192, 128, 64): "Child",
    (192, 192, 128): "Column_Pole",
    (64, 64, 128): "Fence",
    (128, 0, 192): "LaneMkgsDriv",
    (192, 0, 64): "LaneMkgsNonDriv",
    (128, 128, 64): "Misc_Text",
    (192, 0, 192): "MotorcycleScooter",
    (128, 64, 64): "OtherMoving",
    (64, 192, 128): "ParkingBlock",
    (64, 64, 0): "Pedestrian",
    (128, 64, 128): "Road",
    (128, 128, 192): "RoadShoulder",
    (0, 0, 192): "Sidewalk",
    (192, 128, 128): "SignSymbol",
    (128, 128, 128): "Sky",
    (64, 128, 192): "SUVPickupTruck",
    (0, 0, 64): "TrafficCone",
    (0, 64, 64): "TrafficLight",
    (192, 64, 128): "Train",
    (128, 128, 0): "Tree",
    (192, 128, 192): "Truck_Bus",
    (64, 0, 64): "Tunnel",
    (192, 192, 0): "VegetationMisc",
    (0, 0, 0): "Void",
    (64, 192, 0): "Wall",
}

# 合并为 11 类 (常用设置) + Void(忽略)
# 参考 SegNet 论文的类别合并方案
CAMVID_11_CLASSES = {
    0: "Sky",
    1: "Building",
    2: "Pole",          # Column_Pole
    3: "Road",
    4: "Sidewalk",      # Pavement
    5: "Tree",          # Tree + VegetationMisc
    6: "SignSymbol",
    7: "Fence",
    8: "Car",           # Car + SUVPickupTruck + Truck_Bus + Train
    9: "Pedestrian",    # Pedestrian + Child
    10: "Bicyclist",
    255: "Void",        # 忽略
}

# 11 类颜色 (用于可视化)
CAMVID_11_COLORS = np.array([
    [128, 128, 128],  # Sky
    [128, 0, 0],      # Building
    [192, 192, 128],  # Pole
    [128, 64, 128],   # Road
    [0, 0, 192],      # Sidewalk
    [128, 128, 0],    # Tree
    [192, 128, 128],  # SignSymbol
    [64, 64, 128],    # Fence
    [64, 0, 128],     # Car
    [64, 64, 0],      # Pedestrian
    [0, 128, 192],    # Bicyclist
], dtype=np.uint8)

# 32类 -> 11类 映射
CLASS_MAP_32_TO_11 = {
    "Sky": 0,
    "Building": 1, "Wall": 1, "Tunnel": 1, "Archway": 1, "Bridge": 1,
    "Column_Pole": 2, "TrafficLight": 2, "TrafficCone": 2,
    "Road": 3, "LaneMkgsDriv": 3, "LaneMkgsNonDriv": 3, "RoadShoulder": 3,
    "Sidewalk": 4, "ParkingBlock": 4,
    "Tree": 5, "VegetationMisc": 5,
    "SignSymbol": 6, "Misc_Text": 6,
    "Fence": 7,
    "Car": 8, "SUVPickupTruck": 8, "Truck_Bus": 8, "Train": 8,
    "OtherMoving": 8, "MotorcycleScooter": 8, "CartLuggagePram": 8,
    "Pedestrian": 9, "Child": 9,
    "Bicyclist": 10,
    "Animal": 255, "Void": 255,
}

# RGB -> 11类 ID 查找表
def _build_rgb_to_id_lut():
    """构建 RGB -> class_id 查找表"""
    lut = np.zeros((256, 256, 256), dtype=np.uint8)
    lut.fill(255)  # 默认为 Void
    for rgb, name in CAMVID_FULL_CLASSES.items():
        class_id = CLASS_MAP_32_TO_11.get(name, 255)
        lut[rgb[0], rgb[1], rgb[2]] = class_id
    return lut

_RGB_TO_ID_LUT = None

def rgb_to_class_id(rgb_mask: np.ndarray) -> np.ndarray:
    """将 RGB 标注转换为类别 ID"""
    global _RGB_TO_ID_LUT
    if _RGB_TO_ID_LUT is None:
        _RGB_TO_ID_LUT = _build_rgb_to_id_lut()
    return _RGB_TO_ID_LUT[rgb_mask[:, :, 0], rgb_mask[:, :, 1], rgb_mask[:, :, 2]]


def class_id_to_rgb(class_mask: np.ndarray) -> np.ndarray:
    """将类别 ID 转换为 RGB 用于可视化"""
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(11):
        rgb[class_mask == class_id] = CAMVID_11_COLORS[class_id]
    return rgb


class CamVidDataset(Dataset):
    """CamVid 数据集"""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_size: tuple[int, int] = (360, 480),  # (H, W)
        augment: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")

        # 图像和标注目录
        self.img_dir = self.root / split
        self.ann_dir = self.root / f"{split}annot"

        # 获取文件列表
        self.images = sorted(self.img_dir.glob("*.png"))
        assert len(self.images) > 0, f"No images found in {self.img_dir}"

        # 图像变换
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图像和标注
        img_path = self.images[idx]
        ann_path = self.ann_dir / img_path.name

        img = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path)  # 灰度图，值为 class_id

        # 转换为 numpy
        img = np.array(img)
        ann = np.array(ann)

        # SegNet-Tutorial 的标注已经是 class_id (0-10, 11=Void)
        # 将 11 (Void) 映射到 255 用于 ignore_index
        ann[ann == 11] = 255

        # 数据增强
        if self.augment:
            img, ann = self._augment(img, ann)
        else:
            img, ann = self._resize_crop(img, ann)

        # 转换为 tensor
        img = self.img_transform(Image.fromarray(img))
        ann = torch.from_numpy(ann).long()

        return img, ann

    def _augment(self, img: np.ndarray, ann: np.ndarray):
        """训练时数据增强"""
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        # 随机缩放 0.5-2.0
        scale = np.random.uniform(0.5, 2.0)
        new_h, new_w = int(h * scale), int(w * scale)
        img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
        ann = np.array(Image.fromarray(ann).resize((new_w, new_h), Image.NEAREST))

        # 随机裁剪
        h, w = img.shape[:2]
        if h < crop_h or w < crop_w:
            # 填充
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            ann = np.pad(ann, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)
            h, w = img.shape[:2]

        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)
        img = img[start_h:start_h + crop_h, start_w:start_w + crop_w]
        ann = ann[start_h:start_h + crop_h, start_w:start_w + crop_w]

        # 随机水平翻转
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            ann = np.fliplr(ann).copy()

        return img, ann

    def _resize_crop(self, img: np.ndarray, ann: np.ndarray):
        """验证/测试时 resize"""
        crop_h, crop_w = self.crop_size
        img = np.array(Image.fromarray(img).resize((crop_w, crop_h), Image.BILINEAR))
        ann = np.array(Image.fromarray(ann).resize((crop_w, crop_h), Image.NEAREST))
        return img, ann


def get_dataloaders(
    root: str | Path,
    batch_size: int = 8,
    crop_size: tuple[int, int] = (360, 480),
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """获取 train/val/test DataLoader"""
    train_ds = CamVidDataset(root, "train", crop_size, augment=True)
    val_ds = CamVidDataset(root, "val", crop_size, augment=False)
    test_ds = CamVidDataset(root, "test", crop_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# 类别数量 (不含 Void)
NUM_CLASSES = 11
