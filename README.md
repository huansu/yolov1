## yolov1-pytorch版本
backbone部分有修改，没有使用原作者的GoogleNet阉割版，参照博主:https://zhuanlan.zhihu.com/p/365788432, 使用Resnet18作为backbone    
本次复现重点在于数据的encode和decode部分, 而非模型骨架，骨架搭建参见DL仓库，有较为详细的注释      
使用Pascal_Voc2007数据集进行训练，train/eval= 0.8/0.2       
权重文件：链接：https://pan.baidu.com/s/1E7_tiNB1RTpkSjpVjqFR3Q?pwd=cd8a  提取码：cd8a
