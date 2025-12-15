from PIL import Image


def _process(img: Image.Image, params):
    # 占位：不做任何识别处理，直接返回原图的副本
    return img.copy()


PROCESSOR = {
    'id': 'digit_recognition',
    'label': '数字识别',
    'description': '占位处理器：数字识别逻辑尚未接入',
    'process': _process,
}