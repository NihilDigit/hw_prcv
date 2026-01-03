# 关键公式

1) IoU / mIoU
- IoU_c = TP_c / (TP_c + FP_c + FN_c)
- mIoU = mean over classes

2) Poly 学习率调度
- lr = base_lr * (1 - iter / max_iter) ^ 0.9
