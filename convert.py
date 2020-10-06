import torch
import torchsummary
from MobileNetV2_dynamicFPN import MobileNetV2_dynamicFPN
from centerface import Centerface

def convert_to_onnx(model, onnx_name):
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    torch_out = model(x)
    torch.onnx.export(model,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                onnx_name,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=12,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['heatmap', 'scale', 'offset', 'landmarks'], # the model's output names
                # dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                #                 'conf' : {0 : 'batch_size'},
                #                 'loc' : {0 : 'batch_size'}})
    )
    print(f'saved onnx model: {onnx_name}')


base_net = Centerface()
base_net.eval()
convert_to_onnx(base_net, 'my_ctf.onnx')
# print(base_net)
# torch.save(base_net.state_dict(), 'mb_fpn.pt')
# model = torch.load('model_best.pth', map_location='cpu')
# print(type(model))
# torchsummary.summary(model=model, input_size=(1, 3, 640, 640))
# print(model)
# torchsummary.summary(base_net, input_size=(3, 640, 640))
# x = torch.randn(1,3,640,640)
# y = base_net(x)



