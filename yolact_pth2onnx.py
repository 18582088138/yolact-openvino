import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import torch.nn as nn
from yolact import Yolact
from pathlib import Path
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
import cv2
# device = 'cpu'
device = 'cuda'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dir_path = os.path.dirname(__file__)
model_path = "weights/yolact_plus_resnet50_54_800000.pth"
# state_dict  = torch.load(model_path)
# print(list(state_dict.keys()))


pth_path = os.path.join(dir_path,model_path)
img_path = os.path.join(dir_path,"img")
yolact_net = Yolact()
# yolact_net = nn.DataParallel(yolact_net)
# yolact_net.load_weights(pth_path)
# # yolact_net.load_state_dict({
# #     k.replace('module.',''):v for k,v in torch.load(pth_path).items()}) 

yolact_net = yolact_net.to(device)
print(pth_path)
# print(yolact_net)
batch_size = 1  #批处理大小
input_shape = (3, 550, 550)   #输入数据,改成自己的输入shape

# #set the model to inference mode
yolact_net.eval()
	# 目的ONNX文件名
# for p in Path(img_path).glob("*"):
#     path = str(p)
#     print(path)
export_onnx_file = "./yolact_plus_resnet50.onnx"	
path = "data/yolact_example_0.png"
frame = torch.from_numpy(cv2.imread(path)).cuda().float()
batch = FastBaseTransform()(frame.unsqueeze(0))
print("=batch=",type(batch),batch.shape)
preds = yolact_net(batch)
# print("=preds=",preds)
# torch.onnx.export(yolact_net,
#                 batch,
#                 export_onnx_file,
#                 export_params=True,
#                 keep_initializers_as_inputs=True,
#                 opset_version=11,
#                     # 关闭检查，不然可能会报错DCNv2
#                 enable_onnx_checker=False)
torch.onnx.export(yolact_net,
                batch,
                export_onnx_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                opset_version=11)

print("Success")