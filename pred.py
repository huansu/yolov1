import torch
import cv2
import numpy as np

from config import transforms
from YOLO import YoloBody
from tools import decode_boxes, postprocess, vis


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YoloBody(num_classes=20)
    model.to(device)
    model.eval()

    img = cv2.imread("data/000372.jpg")
    image = transforms(image=img)["image"]
    input_size = 416
    w, h = image.shape[1], image.shape[2]
    x = image.permute(0, 1, 2)
    x = x.unsqueeze(0).to(device)
    path_checkpoint = "yolo_64.8_69.2_71.8_73.3.pth"
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['net'])

    with torch.no_grad():
        conf_pred, cls_pred, txtytwth_pred = model(x)
        # batch size = 1
        # [B, H*W, 1] -> [H*W, 1]
        conf_pred = torch.sigmoid(conf_pred)[0]
        # [B, H*W, 4] -> [H*W, 4], 并做归一化处理
        bboxes = torch.clamp((decode_boxes(pred=txtytwth_pred, input_size=input_size, device=device) / torch.tensor(input_size))[0], 0., 1.)
        # [B, H*W, 1] -> [H*W, num_class]，得分=<类别置信度>乘以<objectness置信度>
        scores = (torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred)

        # 将预测放在cpu处理上，以便进行后处理
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # 后处理
        bboxes, scores, cls_inds = postprocess(bboxes, scores)

        scale = np.array([[w, h, w, h]])
        bboxes = bboxes * scale

        # 可视化检测结果
        image = image.squeeze()
            # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
        image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()

        width, heigth = img.shape[1], img.shape[0]
        img_processed = vis(image, bboxes, scores, cls_inds, width, heigth)
        cv2.imshow('detection', img_processed)
        cv2.waitKey(0)