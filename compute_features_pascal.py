import torchvision.transforms as transforms
from data.pascal_voc import PASCALVOC
from torch.utils.data import DataLoader
from models.resnet_wider import resnet50x1
import torch

transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
dataset = PASCALVOC(transform=transform)
dataloader = DataLoader(dataset, num_workers=16,
            batch_size=128, pin_memory=True, drop_last=False, shuffle=False)
model = resnet50x1()['backbone'].cuda()

labels = []
features = []
for batchNum, batch in enumerate(dataloader):
    images, curLabels = batch['image'].cuda(), batch['target'].cuda()
    curFeatures = model(images)
    print(curLabels.squeeze().shape)
    labels.extend(curLabels.squeeze().tolist())
    for i in range(curFeatures.shape[0]):
        print(curFeatures[i].shape)
        features.append(curFeatures[i].tolist())

dataDict = {
    'features': torch.Tensor(features),
    'labels': torch.Tensor(labels)
}

torch.save(dataDict, 'pascalBndBoxFeatures.pth.tar')