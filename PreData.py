# 参考https://blog.zhujian.life/posts/5a56cd45.html
import os
import xml.etree.cElementTree as ET
import csv
from PIL import Image

from config import class_names

def Convert(width, heigth, xmin, ymin, xmax, ymax):
    # 对称填充，改变bounding box的位置，并且对数据进行归一化
    if width >= heigth:
        coefficient = 416 / width
        dif = (416 - (heigth * coefficient)) / 2
        xmin = (xmin * coefficient) / 416
        xmax = (xmax * coefficient) / 416
        ymin = (ymin * coefficient)
        ymax = (ymax * coefficient)
        ymin = (ymin + dif) / 416
        ymax = (ymax + dif) / 416



    elif width < heigth:
        coefficient = 416 / heigth
        dif = (416 - (width * coefficient)) / 2
        ymax = (ymax * coefficient) / 416
        ymin = (ymin * coefficient) / 416
        xmin = (xmin * coefficient)
        xmax = (xmax * coefficient)
        xmin = (xmin + dif) / 416
        xmax = (xmax + dif) / 416

    return xmin, ymin, xmax, ymax

def read_xml(xml_path):
    global txt, img_width, img_heigth
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    LabelsFile = "VOC2007/Labels/"
    # 遍历根节点下所有节点，查询文件名和目标坐标
    for child_node in root:
        if 'filename'.__eq__(child_node.tag):
            img_name = child_node.text
            num, res = os.path.splitext(img_name)
            txt = open('{0}{1}.txt'.format(LabelsFile, num), mode='w+')
        if 'size'.__eq__(child_node.tag):
            if 'width'.__eq__(child_node):
                img_width = child_node[0].text
            if 'heigth'.__eq__(child_node):
                img_heigth = child_node[1].text
        if 'object'.__eq__(child_node.tag):
            obj_name = ''
            for obj_node in child_node:
                if 'name'.__eq__(obj_node.tag):
                    obj_name = obj_node.text
                    class_idx = class_names.index(obj_name)
                    txt.write('{0} '.format(class_idx))

                if 'bndbox'.__eq__(obj_node.tag):
                    node_bndbox = obj_node

                    node_xmin = node_bndbox[0].text
                    node_ymin = node_bndbox[1].text
                    node_xmax = node_bndbox[2].text
                    node_ymax = node_bndbox[3].text

                    node_xmin, node_ymin, node_xmax, node_ymax = Convert(int(img_width),
                                                                         int(img_heigth),
                                                                         int(node_xmin),
                                                                         int(node_ymin),
                                                                         int(node_xmax),
                                                                         int(node_ymax))
                    txt.write('{0} {1} {2} {3}\n'.format(node_xmin,node_ymin,node_xmax,node_ymax))
    txt.close()
    return None

if __name__ == "__main__":
    csv_file = open(r'VOC2007/train.csv', 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(csv_file)
    csv_write.writerow(["ImagePath", "LabelPath"])
    LabelPath = "VOC2007/Annotations/"
    ImgPath = "VOC2007/JPEGImages/"

    LabelList=os.listdir(LabelPath)
    for label in LabelList:
        num, res = os.path.splitext(label)
        csv_write.writerow([num+".jpg", num+".txt"])
        Img = os.path.join(ImgPath, num+".jpg")
        img = Image.open(Img)
        wid = img.width
        heigth = img.height
        path = os.path.join(LabelPath,label)
        print(path)
        read_xml(path)
    csv_file.close()
