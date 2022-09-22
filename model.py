#必要モジュールの読み込み
import torch, torchvision
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.nn import functional as F



#デバイスの作成
def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")
device = get_device(use_gpu=True)

#モデルの作成
model = torchvision.models.resnet152(pretrained=True).to(device)  
with open("imagenet_classes.txt") as f: 
    classes = [line.strip() for line in f.readlines()]
    
#model = torchvision.models.efficientnet_v2_l(weights='IMAGENET1K_V1').to(device)  
#with open("imagenet_classes.txt") as f: 
    #classes = [line.strip() for line in f.readlines()]
    
#model = torchvision.models.convnext_large(weights='IMAGENET1K_V1').to(device)  
#with open("imagenet_classes.txt") as f: 
    #classes = [line.strip() for line in f.readlines()]
        

#前処理（トランスフォーム作成）
def predict(img):
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                        )
                                    ])

# モデルへの入力
    img = transform(img)
    x = torch.unsqueeze(img, 0).to (device)  # バッチ対応

# 予測
    model.eval()
    y = model(x)

# 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
