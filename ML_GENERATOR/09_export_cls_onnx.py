import torch, torchvision
from torch import nn

OUT_ONNX = "cls_best.onnx"
IN_SIZE  = 64
NCLS     = 2

class Head(nn.Module):
    def __init__(self, in_feat, ncls):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feat, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, ncls)
        )
    def forward(self, x): return self.mlp(x)

def make_model():
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    in_feat = m.fc.in_features
    m.fc = Head(in_feat, NCLS)
    return m

if __name__ == "__main__":
    ckpt = torch.load("cls_best.pth", map_location="cpu")
    model = make_model()
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1,3,IN_SIZE,IN_SIZE)
    torch.onnx.export(
        model, dummy, OUT_ONNX, opset_version=13,
        input_names=["images"], output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}
    )
    print("✔ ONNX uložen:", OUT_ONNX)
