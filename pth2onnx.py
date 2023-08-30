import torch
import onnx
from model import MyVGG_deep

pth_path = r'F:\Artificial_neural_Network\yolov8-main\weights\yolov8l.pt'
model = torch.load(pth_path)  # 加载pytorch模型
# model = MyVGG_deep(48, 1)
# model.load_state_dict(torch.load(pth_path))
model.cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)
model.eval()  # 进入评估模式
x = torch.randn(1, 3, 64, 64, device=device)  # torch.randn(batch_size, 通道数， 图片尺寸）
export_onnx_file = 'F:\Artificial_neural_Network\yolov8-main\weights\onnx\yolov8l.onnx'
torch.onnx.export(model,
                  x,
                  export_onnx_file,
                  opset_version=11,
                  input_names=['input'],
                  output_names=['output'])
print('onnx生成完毕')
