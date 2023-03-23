import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import math
from tqdm import tqdm

from transforms import train_transform, train_transform_cuda, val_transform, val_transform_cuda
from config import TrainParams
from OASIS_dataset import get_train_val_test_Dataloaders
from model.HighResNet.highresnet import HighResNet
from model.UNet.unet import UNet3D

def train_model(model):
    
    NUM_CLASSES = TrainParams.OUT_CHANNELS

    if TrainParams.BACKGROUND_AS_CLASS: NUM_CLASSES += 1
    writer = SummaryWriter("runs")

    if model=="highresnet":
        model = HighResNet(in_channels=TrainParams.IN_CHANNELS , num_classes= NUM_CLASSES)
        model_name = "HighResNet"
    if model=="unet":
        model = UNet3D(in_channels=TrainParams.IN_CHANNELS , num_classes= NUM_CLASSES)
        model_name = "UNet"
    
    train_transforms = train_transform
    val_transforms = val_transform
    print('cuda:{}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available() and TrainParams.TRAIN_CUDA:
        model = model.cuda()
        train_transforms = train_transform_cuda
        val_transforms = val_transform_cuda 
    elif not torch.cuda.is_available() and TrainParams.TRAIN_CUDA:
        print('cuda not available! Training initialized on cpu ...')

    train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms,\
                                                                            test_transforms= val_transforms)


    #criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([TrainParams.BCE_WEIGHTS[1]/TrainParams.BCE_WEIGHTS[0]],device='cuda:0'))
    optimizer = Adam(params=model.parameters(), lr=TrainParams.LEARNING_RATE)

    min_valid_loss = math.inf
    print('Training Started..........')
    for epoch in range(TrainParams.TRAINING_EPOCH):
        
        train_loss = 0.0
        model.train()
        for data in tqdm(train_dataloader):
            image, ground_truth = data['image'], data['label']
            optimizer.zero_grad()
            target = model(image)
            #print('Target shape:{}, ground truth shape:{}'.format(target.shape, ground_truth.shape))
            loss = criterion(target, ground_truth)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        valid_loss = 0.0
        model.eval()
        for data in val_dataloader:
            image, ground_truth = data['image'], data['label']
            
            target = model(image)
            loss = criterion(target,ground_truth)
            valid_loss += loss.item()
            
        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
        
        if min_valid_loss > (valid_loss/len(val_dataloader)):
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss/len(val_dataloader):.6f}) \t Saving The Model')
            min_valid_loss = valid_loss/len(val_dataloader)
            # Saving State Dict
            torch.save(model.state_dict(), f'model/{model_name}/checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

        if epoch == (TrainParams.TRAINING_EPOCH-1):
            torch.save(model.state_dict(), f'model/{model_name}/checkpoints/epoch{epoch}_Final.pth')
    writer.flush()
    writer.close()