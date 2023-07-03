import pandas as pd
import torch
from torchvision import datasets, models, transforms
from PIL import Image
from torch import nn
from model import Net
import torch.utils.data as data
import config
import randomSampler
from collections import OrderedDict
from dataset2 import getDataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

model = Net()

model.load_state_dict(torch.load('./trained_models/all2_224.pth',
                                map_location='cuda:0'), strict=False)


if __name__ == '__main__':
    feature_list = []
    img_size = 224
    is_training = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        model.cuda()
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset1 = getDataset(path='CBF', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)
        person_list = dataset8.person_name
        dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=0,
                                     drop_last=False, pin_memory=True)
        dataloader2 = DataLoader(dataset2, batch_size=1, num_workers=0,
                                     drop_last=False, pin_memory=True)
        all_val = zip(enumerate(dataloader1),
                      enumerate(dataloader2))
        for (i, (imgs1, labels1)), (_, (imgs2, labels2))in all_val:
            x1 = imgs1.to(device)
            x2 = imgs2.to(device)

            pre_y = model(x1, x2)
            p = (pre_y.tolist())[0]
            l = (labels1.tolist())[0]
            p.insert(0, l)
            feature_list.append(p)

    data = pd.DataFrame(feature_list, index=person_list)
    data.to_csv('./DeepFeature.csv', index= True)