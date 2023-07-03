import os
import random
import sys

import numpy as np
import pandas as pd
import thop
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms

from center_loss import CenterLoss
from dataset2 import getDataset
from f_loss import FocalLoss
from model import Net_one, ConvMixer, ConvNet, inc, alx, vgg, ConvNet2, Net_one_test
from utils import parse_acc_from_classifaction_report

max_epoch = 70
f1_list = []

# net_name = 'res18_ty-5'


def init_seed(seed):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def train(sq, img_size, model_type, net_name):
    # sys.stdout = open(f'./result/{model_type}/{net_name}_{sq}_{img_size}.log',
    #                   mode='w', encoding='utf-8')
    if img_size == 128:
        batch_size = 256
    elif model_type == 'vgg':
        batch_size = 8
    else:
        batch_size = 128
    # 加载训练集
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.Normalize([0.5], [0.5])
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset1 = getDataset(path=sq, img_size=img_size,
                                transform=train_transforms, is_training=True)
    print(train_dataset1.__len__())
    val_dataset1 = getDataset(path=sq, img_size=img_size,
                              transform=val_transforms, is_training=False)
    print(val_dataset1.__len__())
    train_dataloader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True, num_workers=0,
                                   drop_last=False, pin_memory=True)
    val_dataloader1 = DataLoader(val_dataset1, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, pin_memory=True)

    # 设置GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    modelName = str(img_size) + 'x' + str(img_size)

    if model_type == 'res18':
        model = Net_one(num_class=2)
    # init_network(model)
    # model = Net_one(num_class=2).to(device)
    model.to(device)
    x = torch.randn(32, 1, img_size, img_size).cuda()
    flops, params = thop.profile(model.to("cuda"), inputs=(x,))
    print("  %s   | %s | %s" % ("Model", "Params(Mb)", "FLOPs(Mb)"))
    print("%s |    %.2f    | %.2f" % (modelName, params / (1024 ** 2), flops / (1024 ** 2)))

    # 设置优化器
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0.000,
                                 amsgrad=False
                                 )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.99)

    # eta_min = 0.05 * (0.1 ** 3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 0.0001, -1)

    # 定义损失函数
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = CenterLoss(num_classes=2, feat_dim=1000)
    # criterion3 = FocalLoss()
    # criterion2 = FocalLoss()
    # criterion = nn.NLLLoss()

    # 模型验证
    best_f1 = 0
    # 训练
    for epoch in range(0, max_epoch):
        best_model = model
        model.train()
        train_acc = []
        pred_list = []
        label_list = []
        loss = 0
        for (i, (imgs1, labels1)) in enumerate(train_dataloader1):
            x = imgs1.to(device)
            y = labels1.to(device)
            feat, pre_y = model(x)
            # pre_y = model(x)
            loss1 = criterion1(pre_y, y)
            loss2 = criterion2(feat, y)
            # loss3 = criterion3(pre_y, y)
            if i % 100 == 0:
                # print(f'epoch:{epoch}, loss:{loss1.item() + (loss2.item()) + (loss3.item())}')
                print(f'epoch:{epoch}, loss:{loss1.item()}')
            optimizer.zero_grad()
            (loss1 + loss2).backward()
            # loss1.backward()
            optimizer.step()

            ret, predictions = torch.max(pre_y.data, 1)
            correct_counts = predictions.eq(y.data.view_as(predictions))
            # print(correct_counts)
            tacc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc.append(tacc)
        scheduler.step()

        # 验证
        if (epoch + 1) % 1 == 0:
            model.eval()
            pred_list = []
            label_list = []
            val_acc = []
            for (i, (imgs1, labels1)) in enumerate(val_dataloader1):
                x = imgs1.to(device)
                y = labels1.to(device).item()
                y_ = labels1.to(device)
                feat, pre = model(x)
                # pre = model(x)
                pred_cls = torch.argmax(pre).item()
                pred_list.append(pred_cls)
                label_list.append(y)
                ret, predictions = torch.max(pre.data, 1)
                correct_counts = predictions.eq(y_.data.view_as(predictions))
                vaacc = torch.mean(correct_counts.type(torch.FloatTensor))
                val_acc.append(vaacc)
            # print("*" * 30)
            # print(pred_list)
            # print("*" * 30)
            report = classification_report(label_list, pred_list, labels=[0, 1],
                                           target_names=['GBM', 'SBM'])
            f1, vacc = parse_acc_from_classifaction_report(report)
            f1_list.append(f1)
            print('train acc:{:.4f}, val acc:{:.4f}'.format((sum(train_acc) / len(train_acc)),
                                                            (sum(val_acc) / len(val_acc))))
            # print('val auc:{:.4f}'.format(roc_auc_score(label_list,prob_list)))
            print("*" * 30)

            del pred_list, label_list
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                print(report)
        if epoch == max_epoch - 1:
            torch.save(best_model.state_dict(), f'./trained_models/{sq}_{net_name}_{img_size}.pth')
    df = pd.DataFrame(data=f1_list)
    df.to_csv(f'./trained_models/{net_name}_{sq}_{img_size}.csv', mode="a", encoding="utf_8_sig")


if __name__ == "__main__":

    torch.cuda.empty_cache()
    init_seed(1)
    train('T1WI')