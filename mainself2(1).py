import requests
import time
from plc_connect import plc_db
from wlkata_mirobot import WlkataMirobot
import moveSelf
import maduoXYZ
from realsense_depth import *
import cv2 as cv
import visualSignal
import ast  # 新增：解析字典字符串必需
import os   # 新增：创建目录/路径拼接必需（原代码用了os但未导入）

# 实例化 arm 对象
arm = WlkataMirobot()
# 机械臂初始化（必须）
arm.home()
# 实例化 PLC 对象
PLC = plc_db()

# 连接 plc，直到连接成功
while True:
    message_plc = 'connect plc ok' if PLC.connect() else 'connect plc fail'
    if message_plc == 'connect plc ok':
        break

# 云平台API地址（根据实际接口调整）
CLOUD_API_URL = "http://192.168.40.49:5401"
UPLOAD_ENDPOINT = f"{CLOUD_API_URL}/upload"  # 云平台接收图片的接口
RESULT_ENDPOINT = f"{CLOUD_API_URL}/result"  # 云平台返回结果的接口

# 结果保存根目录（确保真实运行时目录存在）
SAVE_ROOT = "./recognition_results"
os.makedirs(SAVE_ROOT, exist_ok=True)  # 新增：自动创建目录，避免保存失败

# 补全真实运行所需的时间戳函数（如果主程序已有可忽略）
def get_timestamped_filename(prefix, ext):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{ext}"

def visualRecognition():
    time.sleep(2)
    dc = DepthCamera()
    ret, depth_frame, color_frame = dc.get_frame()

    print(f"相机获取帧：ret={ret}")
    if not ret:
        print("警告：未获取到相机帧，跳过保存")
        return None, None, None

    # 1. 裁剪感兴趣区域（保持原有逻辑）
    color_frame_belt = color_frame[178:310, 258:400]

    # 2. 临时保存图像（用于上传）
    temp_filename = get_timestamped_filename("temp_upload", "jpg")
    temp_image_path = os.path.join(SAVE_ROOT, temp_filename)
    try:
        cv.imwrite(temp_image_path, color_frame_belt)
        print(f"临时图像已保存至：{temp_image_path}")
    except Exception as e:
        print(f"❌ 临时图像保存失败：{e}")
        return None, None, None

    # 3. 上传图像到云平台（修正部分）
    uploaded_filename = None
    try:
        with open(temp_image_path, "rb") as f:
            # 关键修正：使用动态生成的temp_filename作为上传文件名
            # 确保与保存的文件名一致，且参数名"file"与app.py匹配
            files = {"file": (temp_filename, f, "image/jpeg")}
            response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=30)
            response.raise_for_status()  # 触发HTTP错误（如500）
            upload_result = response.json()

            # 容错：判断上传成功的字段（匹配云平台返回）
            if not upload_result.get("success", False):
                err_msg = upload_result.get("message", "未知错误")
                print(f"❌ 云平台上传失败：{err_msg}")
                return None, None, None

            uploaded_filename = upload_result.get("filename")
            if not uploaded_filename:
                print(f"❌ 云平台未返回文件名，上传失败")
                return None, None, None

            print(f"✅ 图像上传成功，文件名：{uploaded_filename}")
    except requests.exceptions.Timeout:
        print(f"❌ 上传请求超时（30秒），请检查云平台网络")
        return None, None, None
    except requests.exceptions.ConnectionError:
        print(f"❌ 云平台连接失败，请检查{UPLOAD_ENDPOINT}是否可达")
        return None, None, None
    except requests.exceptions.HTTPError as e:
        # 捕获并显示具体的HTTP错误信息（方便调试）
        print(f"❌ 上传请求HTTP错误：{str(e)}")
        return None, None, None
    except Exception as e:
        print(f"❌ 上传请求失败：{str(e)}")
        return None, None, None


    # 4. 轮询云平台获取解析结果（最多等待30秒，每2秒查一次）
    max_retries = 15
    retry_count = 0
    result_data = None
    while retry_count < max_retries:
        try:
            params = {"filename": uploaded_filename}  # 匹配云平台的参数名
            response = requests.get(RESULT_ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
            result_data = response.json()

            # 检查结果是否就绪
            if result_data.get("ready", False):
                print("✅ 云平台返回解析结果")
                break

            print(f"⏳ 等待解析结果（{retry_count + 1}/{max_retries}）")
            retry_count += 1
            time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"❌ 结果查询超时（10秒）")
            retry_count += 1
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"❌ 云平台连接失败，请检查{RESULT_ENDPOINT}是否可达")
            retry_count += 1
            time.sleep(2)
        except Exception as e:
            print(f"❌ 结果查询失败：{str(e)}")
            retry_count += 1
            time.sleep(2)

    if not result_data or not result_data.get("ready"):
        print("❌ 超时未获取到解析结果（30秒）")
        return None, None, None

    # 5. 解析云平台返回的结果（匹配真实测试验证的格式：英文冒号+自定义字段）
    raw_content = result_data.get("content", "").strip()  # 获取云平台返回的3行文本
    print(f"📝 云平台返回原始结果：\n{raw_content}")

    # 初始化变量（默认值）
    out = None  # 识别的数字（如7）
    conf = None  # 置信度（如0.98）
    shape_type = None  # 分类结果（如"奇"）
    # 兼容原代码的shapes字典（若后续不需要可删除，这里保留避免报错）
    shapes = {"triangle": 0, "rectangle": 0, "polygons": 0, "circles": 0}

    # 按行分割解析（严格匹配你的3行格式）
    lines = [line.strip() for line in raw_content.split("\n") if line.strip()]
    for line in lines:
        # 解析「识别的数字」（格式：识别的数字:7）
        if line.startswith("识别的数字:"):
            num_str = line.split(":", 1)[1].strip()
            if num_str.isdigit():
                out = int(num_str)
            else:
                print(f"⚠️ 识别数字格式错误：{num_str}（应为整数）")
        
        # 解析「置信度」（格式：置信度为:0.98）
        elif line.startswith("置信度为:"):
            conf_str = line.split(":", 1)[1].strip()
            try:
                conf = float(conf_str)
            except ValueError:
                print(f"⚠️ 置信度格式错误：{conf_str}（应为小数）")
        
        # 解析「分类结果」（格式：分类结果：奇）
        elif line.startswith("分类结果："):  # 注意是中文冒号「：」，和前两个英文冒号区分
            shape_type = line.split("：", 1)[1].strip()  # 用中文冒号分割
            # 统一分类结果格式（可选：避免大小写/空格问题）
            shape_type = shape_type.replace(" ", "").lower()

    # 校验解析结果
    if out is None or conf is None or shape_type is None:
        print(f"❌ 解析失败！原始内容：\n{raw_content}")
        print(f"当前解析结果：数字={out}，置信度={conf}，分类={shape_type}")
        return None, None, None, None

    # 5. 保存结果到txt文件（按你的格式保存，包含置信度）
    txt_filename = get_timestamped_filename("recognition_result", "txt")
    txt_save_path = os.path.join(SAVE_ROOT, txt_filename)
    try:
        # 保持和云平台一致的格式保存
        result_content = (
            f"识别的数字:{out}\n"
            f"置信度为:{conf:.2f}\n"
            f"分类结果：{shape_type}"
        )
        with open(txt_save_path, 'w', encoding='utf-8') as f:
            f.write(result_content)
        print(f"✅ 识别结果已保存至：{txt_save_path}")
    except Exception as e:
        print(f"❌ 写入结果文件时出错：{str(e)}")

    # 6. 返回结果：shapes（兼容原代码）、shape_type（分类结果）、out（数字）、conf（置信度）
    print(f"✅ 视觉识别完成：数字={out}，置信度={conf:.2f}，分类结果={shape_type}")
    return shapes, shape_type, out, conf


def moveEndSignal(PLC):
    end = PLC.read('int', 2)
    while end == 0:
        PLC.write(2, bytearray(b'\x00\n'))
        end = PLC.read('int', 2)
        if end == 10:
            break
    time.sleep(2)
    PLC.write(2, bytearray(b'\x00\x00'))


# 取料坐标点
AList = [-65.5, -197.9, 133.6]
BList = [-17.5, -148.9, 128.6]
CList = [69.3, -138.1, 130.0]
DList = [-140.0, 65.2, 131.0]

# 放料中心坐标
one1 = [219.0, 35.0, 142.0]
two2 = [40.6, 225.0, 138.0]

# 循环读取 plc 信息
while True:
    start = PLC.read('int', 0)
    carryStatu = PLC.read('int', 18)
    visual = PLC.read('bool', 44, 0)
    maduoStart = PLC.read('int', 26)

    # 分拣搬运
    if start == 30 and maduoStart == 0 and visual == False:
        print('搬运信号', start)
        # 分拣 A
        if carryStatu == 10:
            startPoint = AList
            endPoint = one1
        # 分拣 B
        elif carryStatu == 20:
            startPoint = BList
            endPoint = two2
        # 分拣 C
        elif carryStatu == 30:
            startPoint = CList
            endPoint = DList

        # 执行搬运程序
        moveSelf.carry(arm, startPoint, endPoint)
        # 完成搬运程序,给 plc 完成信号
        moveEndSignal(PLC)

    elif start == 0 and visual == True:
        time.sleep(1)
        print('视觉识别信号', visual)
        shapes, shape_type, out = visualRecognition()
        if shape_type == '奇数':  # 001
            print(shape_type)
            visualSignal.visual(PLC)
            visualSignal.circular(PLC)

        elif shape_type == '偶数':
            print(shape_type)
            visualSignal.visual(PLC)
            visualSignal.rectangle(PLC)

        elif shape_type == '零':
            print(shape_type)
            visualSignal.visual(PLC)
            visualSignal.triangle(PLC)

    # 分拣堆垛（调用 stackingXYZ 获取堆垛坐标点）
    elif start == 30 and maduoStart == 50 and visual == False:
        print('分拣堆垛信号', maduoStart)
        # xNumOne, yNumOne, zNumOne 需要读取 plc 获得，分别代表行数，列数，层数
        xNumOne = PLC.read('int', 28)
        yNumOne = PLC.read('int', 30)
        zNumOne = PLC.read('int', 32)
        # xNumTwo, yNumTwo, zNumTwo 需要读取 plc 获得，分别代表行数，列数，层数
        xNumTwo = PLC.read('int', 34)
        yNumTwo = PLC.read('int', 36)
        zNumTwo = PLC.read('int', 38)

        num1 = xNumOne * yNumOne * zNumOne
        num2 = xNumTwo * yNumTwo * zNumTwo

        # ranks: 1 是行优先，2 是列优先, order：1 是 Z 次序，2 是 S 次序
        ranks = PLC.read('int', 46)
        order = PLC.read('int', 48)

        # 分拣
        while (num1 >= 0 and maduoStart == 50) or (num2 >= 0 and maduoStart == 50):
            maduoStart = PLC.read('int', 26)
            if maduoStart == 0:
                print('Stop 分拣堆垛信号', maduoStart)
                break

            carryStatu = PLC.read('int', 18)
            if carryStatu == 10:
                # 确定放物坐标点
                XYZ = [244.0, -6.8, 141.0]
                xyzList = maduoXYZ.getXYZList(ranks, order, XYZ[0], XYZ[1], XYZ[2], xNumOne, yNumOne, zNumOne)
                xyzList = xyzList[::-1]
                xyz = xyzList[num1 - 1]
                start = PLC.read('int', 0)
                print('AList', AList)
                print('xyz', xyz)

                if start == 30:
                    moveSelf.carry(arm, AList, xyz)
                    # 完成搬运程序,给 plc 完成信号
                    moveEndSignal(PLC)
                    num1 = num1 - 1
                    print('num1', num1)

            # 分拣 B
            elif carryStatu == 20:
                # 确定放物坐标点
                XYZ = [54.8, 177.3, 139.0]
                xNumOne, yNumOne, zNumOne = 2, 2, 2
                xyzList = maduoXYZ.getXYZList(ranks, order, XYZ[0], XYZ[1], XYZ[2], xNumTwo, yNumTwo, zNumTwo)
                xyzList = xyzList[::-1]
                xyz = xyzList[num2 - 1]
                start = PLC.read('int', 0)
                if start == 30:
                    moveSelf.carry(arm, BList, xyz)
                    # 完成搬运程序,给 plc 完成信号
                    moveEndSignal(PLC)
                    num2 = num2 - 1
                    print('num2', num2)