from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from models.efficientNet import MyEfficientNet
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import torch
import PIL
import cv2

class SquarePad():
    def __call__(self, image):
        w, h = image.size
        # w, h = image.shape[0], image.shape[1]
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TF.pad(image, padding, 0, padding_mode='edge')

def preprocess_image(img: np.ndarray, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    preprocessing = Compose([
        SquarePad(),
        Resize((224,224)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def preprocess_image_wo_norm(img: np.ndarray, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    preprocessing = Compose([
        SquarePad(),
        Resize((224,224)),
        ToTensor()
    ])
    return preprocessing(img.copy()).unsqueeze(0)

# method = 'AblationCAM'  # Expend times to run 
# method = 'GradCAM'
method = 'GradCAMPlusPlus'
# method = 'ScoreCAM'   # Expend times to run
# method = 'XGradCAM'   # Axiom-based Grad-CAM, 2020
# method = 'EigenCAM'   # 2020, good to use

model = MyEfficientNet()
model_path = "models/net_019.pth"

# img_path = "examples/OK/1_1_C16-G08A001100048.JPG"   # Predict success
# save_path = f"examples/OK/{method}_1_1_C16-G08A001100048_out.jpg"

# img_path = "examples/NG/1_1_C103-G08A002400057.JPG"  # Predict success
# save_path = f"examples/NG/{method}_1_1_C103-G08A002400057_out.jpg"

# img_path = "examples/OK_3/1_2_R255-1204137000410.JPG"  # Misclassified_ok_as_ng
# save_path = f"examples/OK_3/{method}_1_2_R255-1204137000410_out.jpg"

img_path = "examples/OK_2/1_1_IOR2-1203876200272.JPG"  # Misclassified_ok_as_ng
save_path = f"examples/OK_2/{method}_1_1_IOR2-1203876200272_out.jpg"

checkpoint = torch.load(model_path)#, map_location='cuda:0')
state_dict = model.state_dict()
for net_key, ckpt_key in zip(state_dict,checkpoint):
    if 'module.' in net_key and not 'module.' in ckpt_key:
        changeName = True
        break
    else:
        changeName = False
        break
if changeName:
    for weightName in checkpoint:
        netName = 'module.'+ weightName
        state_dict[netName]=checkpoint[weightName]
    model.load_state_dict(state_dict, strict=True)
else:
    model.load_state_dict(checkpoint, strict=True)

target_layers = [model.network._bn1]

rgb_img_np = cv2.imread(img_path, 1)[:, :, ::-1]
rgb_img_np = np.float32(rgb_img_np) / 255
rgb_img = PIL.Image.open(img_path)
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
input_tensor_wo_norm = preprocess_image_wo_norm(rgb_img)

# Construct the CAM object once, and then re-use it on many images:
if method =='GradCAM':
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
elif method =='GradCAMPlusPlus':
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
elif method =='ScoreCAM':
    cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=False)
elif method == 'AblationCAM':
    cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=False)
elif method == 'XGradCAM':
    cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=False)
elif method == 'EigenCAM':
    cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)    

target_category = 0 if "OK" in img_path else 1

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, aug_smooth=True, eigen_smooth=True)

grayscale_cam = grayscale_cam[0, :]
input_tensor_wo_norm = input_tensor_wo_norm.squeeze(0)
input_tensor_wo_norm = input_tensor_wo_norm.permute(1, 2, 0).numpy()
visualization = show_cam_on_image(input_tensor_wo_norm, grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
cv2.imwrite(save_path, cam_image)