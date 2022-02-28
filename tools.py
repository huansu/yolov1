import torch.utils.data
import numpy as np
import cv2

from config import transforms, class_names
from Dataset import YOLODataset

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def getdata(batch_size, train, eval):
    transform = transforms
    dataset = YOLODataset(
        "./VOC2007/train.csv",
        "./VOC2007/JPEGImages/",
        "./VOC2007/Labels/",
        transforms=transform,
    )

    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)
    train_data, eval_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
        pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
        pin_memory=True
    )

    if train:
        return train_loader, len(train_data)
    elif eval:
        return eval_loader, len(eval_data)

# 创建含cell的grid矩阵
def create_grid(input_size, device):
    # 输入图像的宽和高
    w, h = input_size, input_size
    # 特征图的宽和高
    ws, hs = w // 32, h // 32
    # 使用torch.meshgrid函数来获得矩阵G的x坐标和y坐标
    grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
    # 将xy两部分坐标拼在一起，得到矩阵G
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    # 最终G矩阵的维度是[1,HxW,2]
    grid_xy = grid_xy.view(1, hs * ws, 2)

    return grid_xy.to(device)

def decode_boxes(pred, input_size, device):
    """
    input box :  [tx, ty, tw, th]
    output box : [xmin, ymin, xmax, ymax]
    """
    grid_cell = create_grid(input_size, device)
    output = torch.zeros_like(pred)
    # 获取bbox的中心点坐标和宽高
    pred[:, :, :2] = (torch.sigmoid(pred[:, :, :2]) + grid_cell) * 32
    pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

    # 由中心点坐标和宽高获得左上角与右下角的坐标
    output[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
    output[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
    output[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
    output[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2

    return output

def generate_txtytwth(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:4]
    # 计算边界框的中心点
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1e-4 or box_h < 1e-4:
        # print('Not a valid data !!!')
        return False

    # 计算中心点所在的网格坐标
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 计算中心点偏移量和宽高的标签
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    # 计算边界框位置参数的损失权重
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=[]):
    # 必要的参数
    batch_size = len(label_lists)
    w = input_size
    h = input_size
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, 1+1+4+1])

    # 制作训练标签
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[4])
            result = generate_txtytwth(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight


    gt_tensor = gt_tensor.reshape(batch_size, -1, 1+1+4+1)

    return torch.from_numpy(gt_tensor).float()

def postprocess(bboxes, scores):
    """
    bboxes: (HxW, 4), bsize = 1
    scores: (HxW, num_classes), bsize = 1
    """

    cls_inds = np.argmax(scores, axis=1)
    scores = scores[(np.arange(scores.shape[0]), cls_inds)]

    # threshold
    keep = np.where(scores >= 0.5)
    bboxes = bboxes[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    # NMS
    keep = np.zeros(len(bboxes), dtype=np.int)
    for i in range(20):
        inds = np.where(cls_inds == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bboxes = bboxes[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    return bboxes, scores, cls_inds

def Convert(width, heigth, xmin, ymin, xmax, ymax):
    # 对称填充，改变bounding box的位置，并且对数据进行归一化
    if width >= heigth:
        coefficient = 416 / width
        dif = (416 - (heigth * coefficient)) / 2
        xmin = (xmin * coefficient)
        xmax = (xmax * coefficient)
        ymin = (ymin * coefficient)
        ymax = (ymax * coefficient)
        ymin = (ymin + dif)
        ymax = (ymax + dif)



    elif width < heigth:
        coefficient = 416 / heigth
        dif = (416 - (width * coefficient)) / 2
        ymax = (ymax * coefficient)
        ymin = (ymin * coefficient)
        xmin = (xmin * coefficient)
        xmax = (xmax * coefficient)
        xmin = (xmin + dif)
        xmax = (xmax + dif)

    return xmin, ymin, xmax, ymax

def vis(img, bboxes, scores, cls_inds, width, heigth):
    class_name = class_names
    class_colors = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(20)]

    for i, box in enumerate(bboxes):
        cls_indx = cls_inds[i]
        xmin, ymin, xmax, ymax = box
        #xmin, ymin, xmax, ymax = Convert(width, heigth, xmin, ymin, xmax, ymax)
        if scores[i] > 0.01:
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
            mess = '%s' % (class_name[int(cls_indx)])
            cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img

def nms(dets, scores):
    """"Pure Python NMS baseline."""
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算交集的左上角点和右下角点的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算交集的宽高
        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        # 计算交集的面积
        inter = w * h

        # 计算交并比
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 滤除超过nms阈值的检测框
        inds = np.where(ovr <= 0.5)[0]
        order = order[inds + 1]

    return keep