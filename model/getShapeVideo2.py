import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse

# 1. 加载你训练好的模型
current_script_dir = os.path.dirname(os.path.abspath(__file__))
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

def preprocess_image(img):
    """
    【Linux 适配最终版】
    输入：原本是黑字（深色字）
    输出：黑底白字（模型要求的格式）
    逻辑：使用 OTSU 自动寻找阈值 + 颜色反转
    """
    # 1. 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 高斯模糊 (关键)
    # 这一步是为了去除噪点，防止 OTSU 计算的阈值受干扰
    # (5, 5) 是模糊核，必须是奇数
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. OTSU 二值化 + 颜色反转 (核心)
    # threshold 设为 0，因为用了 THRESH_OTSU，OpenCV 会自动找最佳值覆盖掉它
    # THRESH_BINARY_INV 作用：
    #   - 图像中"暗"的像素（你的黑色数字） -> 变白 (255)
    #   - 图像中"亮"的像素（你的背景）     -> 变黑 (0)
    ret, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 打印一下自动计算出的阈值，方便你调试观察
    print(f"【调试】OTSU 自动计算的阈值为: {ret}")

    # 4. 形态学膨胀 (可选，建议保留)
    # 因为二值化后数字可能会变细，膨胀可以让字变粗，利于 YOLO 识别
    # 如果你的数字本身就很粗，可以把 iterations 改为 0 或者注释掉
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
    
    # 5. 强制转为 3通道 BGR (YOLO 格式要求)
    preprocessed_img = cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)
    
    return preprocessed_img

def predict_and_classify_silent(image_path):
    try:
        # 读取图片（支持中文路径）
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None, None, None
        print(f"原图通道数：{img.shape[-1]}")  # 应该输出3
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None, None, None

    # --- 图片预处理 ---
    print("正在进行图片预处理...")
    preprocessed_img = preprocess_image(img)
    
    # 额外验证：确保预处理后是3通道
    if len(preprocessed_img.shape) != 3 or preprocessed_img.shape[-1] != 3:
        print("警告：预处理后图像不是3通道！正在强制转换...")
        # 强制转换为3通道（双重保险）
        preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    
    # --- 模型推理 ---
    print("正在进行模型推理...")
    results = model(preprocessed_img, verbose=False)

    # --- 处理结果 ---
    for result in results:
        boxes = result.boxes

        if len(boxes) == 0:
            print("画面中未检测到数字")
            return None, "未检测到数字", 0.0

        # 取置信度最高的结果
        best_conf = 0.0
        best_cls_id = None
        best_category_cn = None
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            if conf > best_conf:
                best_conf = conf
                best_cls_id = cls_id
                best_category_cn = classify_number_logic(cls_id)
        
        return best_cls_id, best_category_cn, best_conf

# --- 主程序 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", required=True, help="结果输出路径")
    parser.add_argument("--threshold", type=int, default=80, help="黑色阈值（0-255）")
    args = parser.parse_args()

    test_image_path = args.input
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
    else:
        # 运行推理
        cls_id, category_cn, conf = predict_and_classify_silent(test_image_path)
        
        # 保存结果
        if cls_id is not None:
            result = (
                f"识别的数字:{cls_id}\n"
                f"置信度为:{conf:.2f}\n"
                f"分类结果：{category_cn}\n"
            )
        else:
            result = (
                f"识别的数字:无\n"
                f"置信度为:0.00\n"
                f"分类结果：{category_cn}\n"
            )
        
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print("处理完成，结果已保存")