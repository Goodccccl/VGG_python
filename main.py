import os

import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torchvision.transforms import transforms

import data
import model

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


if __name__ == '__main__':
    # 数据加载
    img_size = 48
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    # transforms.ToTensor()])
                                    transforms.Normalize(mean=(0.3151, 0.0619, 0.1086), std=(0.2417, 0.0770, 0.2336))])
    # 高数据：transforms.Normalize(mean = (0.28, 0.168, 0.102), std = (0.234, 0.149, 0.233))
    # 自定义数据：transforms.Normalize(mean = (0.2936, 0.057, 0.1002), std = (0.2378, 0.07580, 0.2357))

    root_dir = r"F:\Workprojects\TongFu_Bump\data\perfect_select"
    allDatasets = data.MyData(root_dir, ["error", "normal"], transform)

    img, label, _ = allDatasets[10]

    # length 长度
    # OK_data_size = len(bump_OK_data)
    # NG_data_size = len(bump_NG_data)
    #
    # train_set=Subset(bump_OK_data,range(int(0.8*OK_data_size)))+Subset(bump_NG_data, range(int(0.8 *
    # OK_data_size))) test_set = Subset(bump_OK_data, range(int(0.2 * NG_data_size))) + Subset(bump_NG_data,
    # range(int(0.2 * NG_data_size)))

    k = int(0.8 * len(allDatasets))
    train_set, test_set = random_split(allDatasets, [k, len(allDatasets) - k])

    train_data_size = len(train_set)
    test_data_size = len(test_set)

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True, num_workers=0)

    # 创建模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 重新训练
    myModel = model.MyVGG_deep(img_size, 1)
    # weight_path = r'F:\Workprojects\TongFu_Bump\runs\savedModel_199_accuracy_0.9892473220825195.pth'
    # myModel = torch.load(weight_path)

    myModel.to(device)
    # # 冻结层
    # freeze = 24
    # freeze_layer = [f"features.{x}." for x in range(freeze)]
    # for k, v in myModel.named_parameters():
    #     v.requires_grad = True
    #     if any(x in k for x in freeze_layer):
    #         print(f'freezing {k}')
    #         # v.requi和adam对比res_grad = False

    # 优化器
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)


    # 在指定轮数修改学习率
    lr_steps = [150, 250]
    lr_gamma = 0.01
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=lr_steps,
                                                        gamma=lr_gamma,
                                                        last_epoch=-1)

    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 300
    # 添加tensorboard
    writer = SummaryWriter("./logs_train")

    anomalyThreshold = 0.5

    best_accuracy = 0

    print('当前使用 {} 进行训练。'.format(device))
    for i in range(epoch):
        print("------------第 {} 轮训练开始------------".format(i + 1))
        myModel.train()
        for data in train_loader:
            imgs, targets = data[0], data[1]
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = torch.squeeze(myModel(imgs))
            # loss=model.fcdLossLayer()(outputs,targets)
            loss = nn.BCELoss()(outputs.to(torch.float32), targets.to(torch.float32))

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))  # 这里用到的 item()方法，有说法的，其实加不加都行，就是输出的形式不一样而已
                writer.add_scalar("train_loss", loss.item(), total_train_step)  #

        lr_scheduler.step()

        # 测试步骤开始
        myModel.eval()  # 这两个层，只对一部分层起作用，比如 dropout层；如果有这些特殊的层，才需要调用这个语句
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():  #
            for data in test_loader:  # 在测试集中，选取数据
                imgs, targets = data[0], data[1]
                imgs, targets = imgs.to(device), targets.to(device)
                # targets = torch.squeeze(targets)
                outputs = torch.squeeze(myModel(imgs))
                # print(outputs.size())
                if outputs.size() == torch.Size([]) or targets.size() == torch.Size([]):
                    print("odd data!")
                    continue

                # loss=model.fcdLossLayer()(outputs,targets)
                loss = nn.BCELoss()(outputs.to(torch.float32), targets.to(torch.float32))

                total_test_loss = total_test_loss + loss.item()  # 为了查看总体数据上的 loss，创建的 total_test_loss，初始值是0
                accuracy = ((outputs > anomalyThreshold) == targets).sum()  #
                total_accuracy = total_accuracy + accuracy

        print("当前训练的学习率：{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        print("整体测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))  # 即便是输出了上一行的 loss，也不能很好地表现出效果。
        # 在分类问题上比较特有，通常使用正确率来表示优劣。因为其他问题，可以可视化地显示在tensorboard中。
        # 这里在（二）中，讲了很复杂的，没仔细听。这里很有说法，argmax（）相关的，有截图在word笔记中。
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step = total_test_step + 1

        accuracy_temp = total_accuracy / test_data_size
        output_path = r"runs"
        save_path = os.path.join(output_path, "savedModel_{}_accuracy_{}.pth").format(i, accuracy_temp)
        if i < epoch-10:
            if accuracy_temp >= best_accuracy:
                best_accuracy = accuracy_temp
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                torch.save(myModel, save_path)  # 保存方式一，其实后缀都可以自己取，习惯用 .pth。
                print("模型已更新")

        if i >= epoch-10:
            torch.save(myModel, save_path)
            if i == epoch - 1:
                print("----训练结束----")

    writer.close()
