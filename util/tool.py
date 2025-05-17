
import os,cv2
import matplotlib.pyplot as plt
import numpy as np

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def multiclass_dice(pred, target, eps=1e-6):
    # pred, target: [B, C, H, W] — both one-hot or softmaxed
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    num_classes = pred.shape[1]
    dice_per_class = []

    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c]
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice_c = (2 * inter + eps) / (union + eps)
        dice_per_class.append(dice_c.item())

    mean_dice = sum(dice_per_class) / len(dice_per_class)
    return mean_dice, dice_per_class


def multiclass_iou(pred_cls, target_cls, num_classes, eps=1e-6):
    # pred_cls, target_cls: [B, H, W] with integer values
    iou_per_class = []

    for c in range(num_classes):
        pred_c = (pred_cls == c)
        target_c = (target_cls == c)
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        iou = (inter + eps) / (union + eps)
        iou_per_class.append(iou)

    mean_iou = sum(iou_per_class) / len(iou_per_class)
    return mean_iou, iou_per_class



# def colorize_label(label_tensor, num_classes):
#     """
#     将 [H, W] 的类别标签张量转换为 RGB 彩色图像，便于可视化。
#     """
#     label_np = label_tensor.cpu().numpy()  # [H, W]
#     cmap = plt.get_cmap('nipy_spectral')  # 可选：'tab10', 'Set3', 'nipy_spectral'
#     colored = cmap(label_np / (num_classes - 1))  # [H, W, 4]
#     rgb = (colored[..., :3] * 255).astype(np.uint8)  # 转为 RGB
#     return rgb


def colorize_label(label_tensor, num_classes):
    label_tensor = label_tensor.cpu().numpy().astype(np.uint8)  # 确保是 uint8 类型
    cmap = plt.get_cmap('tab10', num_classes)  # 支持最多10类，可换成其他cmap
    color_img = cmap(label_tensor)[..., :3] * 255  # 去除 alpha 通道并乘255
    return color_img.astype(np.uint8)