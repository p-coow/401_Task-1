import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
# 1. 加载你训练好的模型
# 假设 best.pt 文件在当前目录或指定路径下
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接模型文件的绝对路径（假设模型和脚本在同一目录）
model_path = os.path.join(current_script_dir, "gsv2.pt")

model = YOLO(model_path)

def classify_number_logic(label_id):

    digit = int(label_id)

    if digit == 0:
        return "零"
    elif digit % 2 == 0:
        return "偶"
    else:
        return "奇"


def predict_and_classify_silent(image_path):

    try:
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
    except Exception as e:
        print(f"读取文件错误: {e}")
        return

    # --- 2. 模型推理 ---
    # 注意: 我们不需要可视化，所以设置 verbose=False
    results = model(img, verbose=False)

    # --- 3. 处理结果 ---
    for result in results:
        boxes = result.boxes

        # 如果没有检测到目标
        if len(boxes) == 0:
            print("画面中未检测到数字")
            return

        # 遍历所有检测到的框
        for box in boxes:
            # 获取类别 ID (这是模型预测的数字 0-9)
            cls_id = int(box.cls[0].item())
            # 获取置信度
            conf = box.conf[0].item()

            # 【关键步骤】调用分类逻辑函数获取中文结果
            category_cn = classify_number_logic(cls_id)
            return cls_id, category_cn, conf

# --- 测试部分 ---
if __name__ == '__main__':
    # 请替换为你想要测试的图片路径
    # 支持中文路径
    test_image_path = r'C:\Users\16638\Desktop\123\1.jpg'

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", required=True, help="结果输出路径")
    args = parser.parse_args()

    test_image_path = args.input
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}，请确保模型已训练好并放在正确位置。")
    else:
        # 运行推理
        cls_id, category_cn, conf = predict_and_classify_silent(test_image_path)
        result = (
            f"识别的数字:{cls_id}\n"
            f"置信度为:{conf:.2f}\n"
            f"分类结果：{category_cn}"
            
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print("处理完成，结果已保存")