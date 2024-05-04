import torch

from mit_ub.model.backbone import ViT


class TestViT:

    def test_forward(self):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        nhead = 128 // 16
        model = ViT(3, 128, (16, 16), 3, nhead).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
        assert out.shape[:2] == (1, 128)

    def test_backward(self):
        x = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
        nhead = 128 // 16
        model = ViT(3, 128, (16, 16), 3, nhead).cuda()
        scaler = torch.amp.GradScaler()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
            out = out.sum()
        scaler.scale(out).backward()
