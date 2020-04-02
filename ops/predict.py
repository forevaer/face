import cv2
import torch
from config.config import celoss
from utils.image import draw_points, image_show, normal_image, tensor_image


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return tensor_image(image)


def predict(model, image_path):
    model.eval()
    image = load_image(image_path)
    out_pts, out_cls = model(image)
    out_pts = out_pts.detach().numpy()
    image = normal_image(image)
    if out_cls.argmax(dim=1, keepdim=True).squeeze() == 1:
        image = draw_points(image, out_pts)
    image_show(image)



