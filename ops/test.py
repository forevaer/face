import cv2
import torch
from config import config
from utils.image import draw_points, image_show, normal_image


def test(loader, model):
    device = config.device()
    cls_loss_func = config.celoss
    pts_loss_func = config.mseloss
    model.to(device)
    model.eval()
    for idx, item in enumerate(loader):
        image = item['image']
        points = item['points']
        label = item['label']
        image = image.to(device)
        points = points.to(device)
        label = label.to(device)
        with torch.no_grad():
            out_pts, out_cls = model(image)
            pts_loss = pts_loss_func(out_pts, points)
            cls_loss = cls_loss_func(out_cls, label.view(-1).long())
            print(f'pts_loss : {pts_loss} \t cls_loss : {cls_loss}')
            image = normal_image(image)
            points = points.numpy().reshape((-1, 2))
            out_pts = out_pts.numpy().reshape((-1, 2))
            print(f'classify : {out_cls.argmax(dim=1, keepdim=True).squeeze() == label.item()}')
            if label.item() == 1:
                image = draw_points(image, points, color=(255, 0, 0))
                image = draw_points(image, out_pts, color=(0, 0, 255))
            image_show(image)
