import numpy as np
import torch
from config import config


def train(loader, model):
    device = config.device()
    # loss
    pts_criterion = config.mseloss
    cls_criterion = config.celoss
    # optimizer
    optimizer = config.optimizer(model)
    model.train()
    for epoch_id in range(config.epochs):
        for batch_idx, item in enumerate(loader):
            image = item['image'].to(device)
            points = item['points'].to(device)
            points.requires_grad = True
            label = item['label'].to(device)
            # 清空历史梯度
            optimizer.zero_grad()
            output_pts, output_cls = model(image)
            # 有脸的数据索引
            face_index = label == 1
            # face索引
            face_index = np.squeeze(face_index)
            # non-face索引
            none_index = face_index == False
            # face样本个数
            face_count = face_index.sum().item()
            # non-face样本个数
            none_count = len(none_index) - face_count
            """
                初始化face参数
            """
            loss_face = 0
            predict_face_cls_correct_acc = 1.0
            loss_face_pts = 0
            loss_face_cls = 0
            """
                初始化non-face参数
            """
            loss_none = 0
            loss_none_cls = 0
            predict_none_cls_correct_acc = 1
            """
            face: 
                detect  ： 提供人脸识别损失
                classify： 提供分类识别损失
            non:
                detect  : 不提供人脸识别损失
                classify: 只提供分类识别损失
            """

            if not all(none_index):
                # 计算人脸识别损失
                loss_face_pts = pts_criterion(output_pts[face_index],
                                              points[face_index])
                # 计算分类识别损失，由于分类数据少，为了提高权值，*50
                loss_face_cls = 50 * cls_criterion(
                    output_cls[face_index], label[face_index].view(-1).long())
                # face样本贡献梯度，为两个梯度之和
                loss_face = loss_face_pts + loss_face_cls
                # 分类预测标签
                predict_face_class = output_cls[face_index].argmax(dim=1, keepdim=True)
                # 分类结果对比
                predict_face_cls_correct_acc = \
                    predict_face_class.eq(label[face_index]).sum().item() / face_count

            if not all(face_index):
                # non-face分类损失贡献，提升权重 *50
                loss_none = loss_none_cls = 50 * cls_criterion(
                    output_cls[none_index], label[none_index].view(-1).long())
                # non-face分类预测
                predict_none_cls = output_cls[none_index].argmax(dim=1, keepdim=True)
                # 分类预测对比
                predict_none_cls_correct_acc = \
                    predict_none_cls.eq(label[none_index]).sum().item() / none_count
            # 分类准确率
            predict_cls_acc = \
                (predict_face_cls_correct_acc * face_count +
                 predict_none_cls_correct_acc * none_count) / \
                (face_count + none_count)

            # 总损失贡献， 分类损失权值 *50
            loss = loss_face + 50 * loss_none
            # 损失回传
            loss.backward()
            # 梯度更新
            optimizer.step()
            if batch_idx % config.log_interval == 0:
                print(
                    'epoch: {}\t\t'
                    'loss: {:10.6f}\t\t'
                    'loss_face_pts: {:10.6f}\t\t'
                    'loss_face_cls: {:10.6f}\t\t'
                    'loss_none_cls: {:10.6f}\t\t'
                    'acc: {:10.6f}\t\t'
                    'face_acc: {:10.6f}\t\t'
                    'none_acc: {:10.6f}'.format(
                        epoch_id,
                        loss,
                        loss_face_pts,
                        loss_face_cls,
                        loss_none_cls,
                        predict_cls_acc,
                        predict_face_cls_correct_acc,
                        predict_none_cls_correct_acc
                    )
                )

            if epoch_id % config.save_interval == 0:
                torch.save(model.state_dict(), config.model_path)
