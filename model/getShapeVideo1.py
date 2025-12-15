import torch
import torchvision.transforms as transforms
from PIL import Image
from model import MyLeNet
import cv2 as cv
import numpy as np
import argparse  # 新增：支持命令行参数
import os  # 新增：用于处理路径


class ShapeAnalysis:
    def __init__(self):
        # 保留形状计数核心逻辑
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}
        # 提前模型初始化（移到__init__，避免重复加载）
        self.net = MyLeNet()
        # 关键修改：用绝对路径加载模型
        # 获取当前脚本（getShapeVideo1.py）所在的目录
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接模型文件的绝对路径（假设模型和脚本在同一目录）
        model_path = os.path.join(current_script_dir, "MNistLeNet.pth")
        # 加载模型
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.net.eval()  # 设为评估模式
        # 保留图像预处理逻辑
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def analysis(self, frame):
        """保留核心图像分析与形状判断逻辑"""
        print("start to detect lines...\n")
        
        # 保留图像预处理（灰度化→二值化→腐蚀→膨胀）
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 170, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        binary = 255 - binary  # 反二值化
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv.erode(binary, kernel)  # 腐蚀
        kernel = np.ones((10, 10), np.uint8)
        frame_processed = cv.dilate(erosion, kernel)  # 膨胀
        
        # 保留中间结果保存（便于调试）
        cv.imwrite('1.jpg', frame_processed)
        
        # 保留模型推理逻辑
        frame_pil = Image.open('1.jpg')
        frame_transformed = self.transform(frame_pil)  # [C, H, W]
        frame_input = torch.unsqueeze(frame_transformed, dim=0)  # [N, C, H, W]
        
        with torch.no_grad():  # 禁用梯度计算，加速推理
            outputs = self.net(frame_input)
            predict = torch.max(outputs, dim=1)[1].numpy()
            out = self.classes[int(predict)]
            out = int(out)
            print(f"模型预测结果: {out}")

        # 保留形状类型判断与计数逻辑
        shape_type = ""
        if out == 0:
            self.shapes['triangle'] += 1
            shape_type = "零"
            print("零")
        elif (out % 2) != 0:
            self.shapes['rectangle'] += 1
            shape_type = "奇数"
            print("奇数")
        else:
            self.shapes['circles'] += 1
            shape_type = "偶数"
            print("偶数")

        return self.shapes, shape_type, out


if __name__ == "__main__":
    # 新增：解析命令行参数（适配自动处理脚本）
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", required=True, help="结果输出路径")
    args = parser.parse_args()

    try:
        # 读取输入图片（替换原固定路径，支持动态输入）
        src = cv.imread(args.input)
        if src is None:
            raise Exception(f"无法读取图片: {args.input}")

        # 执行分析（保留核心流程）
        ld = ShapeAnalysis()
        shapes_count, shape_type, detectes_num = ld.analysis(src)

        # 保存结果到输出文件（适配自动处理脚本的结果读取）
        result = (
            f"识别的数字:{detectes_num}\n" #显示模型识别的具体数字
            f"形状计数: {shapes_count}\n" #形状计数
            f"当前形状类型: {shape_type}" #形状类型(零/奇数/偶数)
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print("处理完成，结果已保存")

    except Exception as e:
        # 错误处理：保存错误信息到输出文件
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"处理失败: {str(e)}")
        print(f"处理失败: {str(e)}")
        exit(1)
