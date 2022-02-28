from torch import optim
import torch.utils.data
import time
from tqdm import tqdm

from YOLO import YoloBody
from tools import getdata, gt_creator
from loss import loss

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YoloBody(num_classes=20)
    model.to(device)

    batch_size = 32
    epoch = 200
    lr = 0.001
    #是否加载断点继续训练
    Continue = True

    # 加载数据集
    train_loader, len_train = getdata(batch_size=batch_size, train=True, eval=False)
    eval_loader, len_eval = getdata(batch_size=batch_size, train=False, eval=True)
    # 构建训练优化器
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.8,
                          #weight_decay=5e-4
                          )
    #学习率衰减
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1)

    epoch_train = len_train // batch_size  # 每一训练轮次的迭代次数
    epoch_eval = len_eval // batch_size  # 每一训练轮次的迭代次数

    # 开始训练
    t0 = time.time()
    start_epoch = -1

    if Continue:
        path_checkpoint = "ckpt_best_153_6.863467.pth"  # 断点路径
        checkpoint =torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint["net"])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        for epoch in range(start_epoch + 1, epoch):
            model.train()
            mean_loss = 0
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (images, labels) in enumerate(loop):
                # 将矩阵转换成列表
                targets = [label.tolist() for label in labels]
                # gt_creator的作用
                targets = gt_creator(input_size=416,
                                           stride=32,
                                           label_lists=targets
                                           )
                images = images.to(device)
                targets = targets.to(device)

                # 前向推理，计算损失
                conf_pred, cls_pred, txtytwth_pred = model(images)
                conf_loss, cls_loss, bbox_loss, total_loss = loss(pred_conf=conf_pred,
                                                                        pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=targets
                                                                        )

                # 反向传播和模型更新
                total_loss.backward()
                optimizer.step()
               # lr_scheduler.step()
                optimizer.zero_grad()
                loop.set_postfix(loss=total_loss)
                mean_loss += total_loss
            print("\n train_loss为：{0:.2f}".format(mean_loss / epoch_train))
            checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
              #  'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, 'ckpt_best_{0}_{1:.6f}.pth'.format(epoch, mean_loss / epoch_train))
            print("{0}次训练完成".format(epoch))

            if epoch % 3 == 0:
                model.eval()
                mean_loss = 0
                loop = tqdm(eval_loader, leave=True)
                for batch_idx, (images, labels) in enumerate(loop):
                    # 将矩阵转换成列表
                    with torch.no_grad():
                        targets = [label.tolist() for label in labels]
                        # gt_creator的作用
                        targets = gt_creator(input_size=416,
                                             stride=32,
                                             label_lists=targets
                                             )
                        images = images.to(device)
                        targets = targets.to(device)

                        # 前向推理，计算损失
                        conf_pred, cls_pred, txtytwth_pred = model(images)
                        conf_loss, cls_loss, bbox_loss, total_loss = loss(pred_conf=conf_pred,
                                                                          pred_cls=cls_pred,
                                                                          pred_txtytwth=txtytwth_pred,
                                                                          label=targets
                                                                          )

                        # 计算验证损失
                        loop.set_postfix(loss=total_loss)
                        mean_loss += total_loss
                print("\n val_loss为：{0:.2f}".format(mean_loss / epoch_eval))

    else:
        for epoch in range(start_epoch + 1, epoch):
            model.train()
            mean_loss = 0
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (images, labels) in enumerate(loop):
                # 将矩阵转换成列表
                targets = [label.tolist() for label in labels]
                # gt_creator的作用
                targets = gt_creator(input_size=416,
                                           stride=32,
                                           label_lists=targets
                                           )

                images = images.float().to(device)
                targets = targets.to(device)

                # 前向推理，计算损失
                conf_pred, cls_pred, txtytwth_pred = model(images)
                conf_loss, cls_loss, bbox_loss, total_loss = loss(pred_conf=conf_pred,
                                                                        pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=targets
                                                                        )

                # 反向传播和模型更新
                total_loss.backward()
                optimizer.step()
                #lr_scheduler.step()
                optimizer.zero_grad()
                loop.set_postfix(loss=total_loss)
                mean_loss += total_loss
            print("\n train_loss为：{0:.6f}".format(mean_loss / epoch_train))
            checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                #'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, 'ckpt_best_{0}_{1:.2f}.pth'.format(epoch, mean_loss / epoch_train))
            print("{0}次训练完成".format(epoch))

            if epoch % 3 == 0:
                model.eval()
                mean_loss = 0
                loop = tqdm(eval_loader, leave=True)
                for batch_idx, (images, labels) in enumerate(loop):
                    with torch.no_grad():
                    # 将矩阵转换成列表
                        targets = [label.tolist() for label in labels]
                        # gt_creator的作用
                        targets = gt_creator(input_size=416,
                                             stride=32,
                                             label_lists=targets
                                             )
                        images = images.to(device)
                        targets = targets.to(device)

                        # 前向推理，计算损失
                        conf_pred, cls_pred, txtytwth_pred = model(images)
                        conf_loss, cls_loss, bbox_loss, total_loss = loss(pred_conf=conf_pred,
                                                                          pred_cls=cls_pred,
                                                                          pred_txtytwth=txtytwth_pred,
                                                                          label=targets
                                                                          )

                        # 计算验证损失
                        loop.set_postfix(loss=total_loss)
                        mean_loss += total_loss
                print("\n val_loss为：{0:.2f}".format(mean_loss / epoch_eval))