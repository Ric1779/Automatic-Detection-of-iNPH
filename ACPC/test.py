import torch
from torch.nn import BCEWithLogitsLoss
import math
import glob
from tqdm import tqdm
from transforms import val_transform, val_transform_cuda, train_transform, train_transform_cuda
from dataset import get_train_val_test_Dataloaders
from model.HighResNet.highresnet import HighResNet
from model.UNet.unet import UNet3D
from config import EvaluateParams, TrainParams

from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
from itkwidgets import view
import matplotlib.pyplot as plt


def test_model(model):
    NUM_CLASSES = EvaluateParams.OUT_CHANNELS
    if EvaluateParams.BACKGROUND_AS_CLASS: NUM_CLASSES += 1

    if model=="highresnet":
        model = HighResNet(in_channels=EvaluateParams.IN_CHANNELS , num_classes= NUM_CLASSES)
        model_name = "HighResNet"
        
    if model=="unet":
        model = UNet3D(in_channels=EvaluateParams.IN_CHANNELS , num_classes= NUM_CLASSES)
        model_name = "UNet"

    path = glob.glob(f'model/{model_name}/checkpoints/*_Final.pth')[0]    
    model.load_state_dict(torch.load(path))
    val_transforms = val_transform
    train_transforms = train_transform
    print('cuda:{}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available() and EvaluateParams.TRAIN_CUDA:
        model = model.cuda()
        val_transforms = val_transform_cuda 
        train_transforms = train_transform_cuda
    elif not torch.cuda.is_available() and EvaluateParams.TRAIN_CUDA:
        print('cuda not available! Training initialized on cpu ...')

    train_loader, _, test_loader = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms,\
                                                        test_transforms= val_transforms)
    
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([TrainParams.BCE_WEIGHTS[1]/TrainParams.BCE_WEIGHTS[0]],device='cuda:0'))
    model.eval()
    """
    test_loss = 0
    for data in tqdm(test_loader):
        image, ground_truth = data['image'], data['label']
        target = model(image)
        loss = criterion(target,ground_truth)
        test_loss += loss
    
    print('Test Loss:{}'.format(test_loss/len(test_loader)))

    """
    sample = next(iter(test_loader))
    target = model(sample['image'])
    target = target.cpu()
    target[target>0.7] = 1
    target[target<=0.7] = 0 
    sample['image'] = sample['image'].cpu()
    sample['label'] = sample['label'].cpu()
    label_max = torch.max(sample['label'])
    label_min = torch.min(sample['label'])
    print(f"Max and Min value in label:{label_max},{label_min}")
    image_max = torch.max(sample['image'])
    image_min = torch.min(sample['image'])
    print(f"Max and Min value in image:{image_max},{image_min}")
    sample['image'] = (sample['image']-image_min)
    sample['image'] = sample['image']/torch.max(sample['image'])
    image_max = torch.max(sample['image'])
    image_min = torch.min(sample['image'])
    print(f"Max and Min value in image:{image_max},{image_min}")

    ret = blend_images(image=sample['image'][0], label=target[0], alpha=0.5, cmap="hsv", rescale_arrays=False)
    
    for i in range(5, 7):
        # plot the slice 50 - 100 of image, label and blend result
        slice_index = 3 * i
        plt.figure("blend image and label", (12, 4))
        plt.subplot(1, 3, 1)
        plt.title(f"image slice {slice_index}")
        plt.imshow(sample["image"][0, 0, :, :, slice_index], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label slice {slice_index}")
        plt.imshow(sample["label"][0, 0, :, :, slice_index])
        plt.subplot(1, 3, 3)
        plt.title(f"blend slice {slice_index}")
        # switch the channel dim to the last dim
        plt.imshow(torch.moveaxis(ret[:, :, :, slice_index], 0, -1))
        plt.show()

    