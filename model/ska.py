import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SKA(nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 1. Xử lý w nếu chưa có chiều không gian (ít gặp trong LSNet nhưng cứ để cho chắc)
        if w.dim() == 3:
            w = w.view(B, C, -1, 1, 1)

        # Lấy kernel size
        ks_sq = w.shape[2]
        ks = int(math.sqrt(ks_sq))
        pad = (ks - 1) // 2

        # 2. Unfold x (Cắt ảnh thành các ô)
        # x_unfold shape: [B, C, K*K, H, W]
        x_unfold = F.unfold(x, kernel_size=ks, padding=pad, stride=1)
        x_unfold = x_unfold.view(B, C, ks * ks, H, W)

        # --- ĐOẠN CODE FIX LỖI 128 vs 16 ---
        # Kiểm tra nếu số channel của x và w không khớp (do group convolution)
        # x: [B, 128, ...] vs w: [B, 16, ...]
        if x_unfold.shape[1] != w.shape[1]:
            # Tính tỉ lệ group (ví dụ 128 / 16 = 8)
            groups = x_unfold.shape[1] // w.shape[1]
            # Lặp lại w để khớp với x (Tương đương logic % trong Triton)
            # w.repeat(1, groups, 1, 1, 1) sẽ nhân bản channel lên 8 lần
            w = w.repeat(1, groups, 1, 1, 1)
        # -----------------------------------

        # 3. Nhân và cộng
        out = x_unfold * w
        out = torch.sum(out, dim=2)
        
        return out