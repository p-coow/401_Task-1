from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import io
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import importlib
import pkgutil
import threading
import time
import sys
import subprocess
from typing import Dict, List, Optional, Set

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # lab401/
HTML_DIR = os.path.join(BASE_DIR, 'html')
HTML_FILES_DIR = os.path.join(HTML_DIR, 'html_files')
CSS_DIR = os.path.join(HTML_DIR, 'css_files')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
PROCESSORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processors')

# 确保目录存在
for dir_path in [UPLOAD_DIR, RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 线程安全的变量和锁
queue_lock = threading.Lock()
PROCESSED_FILES: Set[str] = set()
NEW_FILES_QUEUE: List[str] = []
LATEST_IMAGE: Optional[str] = None
LATEST_IMAGE_UPDATED_AT: float = 0.0

# 处理器相关
PROCESSORS: Dict[str, Dict] = {}


def allowed_file(filename: str) -> bool:
    """检查文件是否为允许的类型"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '请求中未包含文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # 生成唯一文件名避免冲突
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
            new_filename = f"{timestamp}_{unique_id}_{name}{ext}"
            
            file_path = os.path.join(UPLOAD_DIR, new_filename)
            file.save(file_path)
            
            # 添加到处理队列
            with queue_lock:
                if new_filename not in PROCESSED_FILES:
                    NEW_FILES_QUEUE.append(new_filename)
                    PROCESSED_FILES.add(new_filename)
            
            return jsonify({
                'success': True,
                'message': '文件上传成功',
                'filename': new_filename,
                'url': f"/uploads/{new_filename}"
            })
        except Exception as e:
            app.logger.error(f"文件上传失败: {str(e)}")
            return jsonify({'success': False, 'message': f'上传失败：{str(e)}'}), 500
    
    return jsonify({
        'success': False,
        'message': f'不支持的文件格式，允许的格式：{ALLOWED_EXTENSIONS}'
    }), 400


def load_processors() -> None:
    """加载所有处理器插件"""
    global PROCESSORS
    PROCESSORS.clear()
    
    if not os.path.isdir(PROCESSORS_DIR):
        app.logger.warning(f"处理器目录不存在: {PROCESSORS_DIR}")
        return
        
    for finder, name, ispkg in pkgutil.iter_modules([PROCESSORS_DIR]):
        try:
            module = importlib.import_module(f'processors.{name}')
            meta = getattr(module, 'PROCESSOR', None)
            
            if meta and all(k in meta for k in ('id', 'label', 'process')):
                PROCESSORS[meta['id']] = meta
                app.logger.info(f"加载处理器成功: {meta['id']}")
            else:
                app.logger.warning(f"处理器 {name} 缺少必要属性")
        except Exception as e:
            app.logger.error(f"加载处理器 {name} 失败: {e}")


def _ensure_result_for(filename: str) -> None:
    """确保处理结果存在"""
    base, _ = os.path.splitext(filename)
    result_name = f"{base}_result.txt"
    result_path = os.path.join(RESULT_DIR, result_name)
    
    if os.path.exists(result_path):
        return
        
    src_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(src_path):
        app.logger.warning(f"源文件不存在: {src_path}")
        return
        
    algo_path = os.path.join(BASE_DIR, 'model', 'getShapeVideo2.py')
    if not os.path.exists(algo_path):
        app.logger.error(f"算法脚本不存在: {algo_path}")
        return
        
    try:
        cmd = [
            sys.executable, 
            algo_path, 
            '--input', os.path.abspath(src_path), 
            '--output', os.path.abspath(result_path)
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        if result.returncode != 0:
            app.logger.error(f"算法执行失败: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        app.logger.error(f"算法执行超时: {filename}")
    except Exception as e:
        app.logger.error(f"处理文件 {filename} 时出错: {str(e)}")


def _background_watch() -> None:
    """后台监控线程，处理新上传的文件"""
    global LATEST_IMAGE, LATEST_IMAGE_UPDATED_AT
    
    while True:
        try:
            # 处理队列中的新文件
            with queue_lock:
                files_to_process = NEW_FILES_QUEUE.copy()
                NEW_FILES_QUEUE.clear()
            
            for filename in files_to_process:
                _ensure_result_for(filename)
                file_path = os.path.join(UPLOAD_DIR, filename)
                
                with queue_lock:
                    LATEST_IMAGE = filename
                    try:
                        LATEST_IMAGE_UPDATED_AT = os.path.getmtime(file_path)
                    except Exception:
                        LATEST_IMAGE_UPDATED_AT = time.time()
            
            time.sleep(1)
        except Exception as e:
            app.logger.error(f"后台监控线程出错: {str(e)}")
            time.sleep(2)


# 初始化已处理文件集合
def init_processed_files() -> None:
    """初始化已处理文件集合"""
    try:
        for name in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, name)
            if os.path.isfile(file_path) and allowed_file(name):
                PROCESSED_FILES.add(name)
        app.logger.info(f"初始化已处理文件数量: {len(PROCESSED_FILES)}")
    except Exception as e:
        app.logger.error(f"初始化已处理文件集合失败: {str(e)}")


# 路由：HTML和静态资源
@app.route('/')
def index():
    return send_from_directory(HTML_FILES_DIR, 'index.html')


@app.route('/css_files/<path:path>')
def serve_css(path):
    return send_from_directory(CSS_DIR, path)


@app.route('/uploads/<path:path>')
def serve_uploads(path):
    return send_from_directory(UPLOAD_DIR, path)


@app.route('/assets/<path:path>')
def serve_assets(path):
    workspace_root = os.path.dirname(os.path.dirname(BASE_DIR))
    assets_dir = os.path.join(workspace_root, 'html', 'assets')
    return send_from_directory(assets_dir, path)


# API路由
@app.route('/latest_image', methods=['GET'])
def latest_image():
    """获取最新上传的图片"""
    with queue_lock:
        fname = LATEST_IMAGE
        updated_at = LATEST_IMAGE_UPDATED_AT
        
    if not fname:
        return jsonify({'success': True, 'ready': False})
        
    return jsonify({
        'success': True, 
        'ready': True, 
        'filename': fname, 
        'url': f"/uploads/{fname}", 
        'updated_at': updated_at
    })


@app.route('/result', methods=['GET'])
def get_result():
    """获取处理结果"""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': '缺少参数: filename'}), 400

    base, _ = os.path.splitext(filename)
    result_name = f"{base}_result.txt"
    result_path = os.path.join(RESULT_DIR, result_name)

    if not os.path.exists(result_path):
        return jsonify({'success': True, 'ready': False})

    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        mtime = os.path.getmtime(result_path)
        return jsonify({
            'success': True,
            'ready': True,
            'filename': result_name,
            'content': content,
            'updated_at': mtime
        })
    except Exception as e:
        app.logger.error(f"读取结果文件失败: {str(e)}")
        return jsonify({'success': False, 'message': f'读取结果失败: {str(e)}'}), 500


@app.route('/processors', methods=['GET'])
def list_processors():
    """列出所有可用的处理器"""
    return jsonify({
        'success': True,
        'processors': [
            {
                'id': p['id'],
                'label': p.get('label', p['id']),
                'description': p.get('description', '')
            } for p in PROCESSORS.values()
        ]
    })


@app.route('/process', methods=['POST'])
def process_image():
    """使用指定处理器处理图片"""
    data = request.get_json(silent=True) or {}
    filename = data.get('filename')
    proc_id = data.get('processor_id')
    params = data.get('params') or {}

    if not filename or not proc_id:
        return jsonify({'success': False, 'message': '缺少必要参数 filename 或 processor_id'})
        
    if proc_id not in PROCESSORS:
        return jsonify({'success': False, 'message': f'未找到处理器: {proc_id}'})

    src_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(src_path):
        return jsonify({'success': False, 'message': '图片不存在'})

    try:
        with Image.open(src_path) as img:
            # 确保图片在处理前被正确加载
            img.load()
            processed = PROCESSORS[proc_id]['process'](img, params)

        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'png'
        out_name = f"{proc_id}_{filename}"
        out_path = os.path.join(UPLOAD_DIR, out_name)
        processed.save(out_path)

        return jsonify({
            'success': True,
            'message': '图片处理成功',
            'url': f"/uploads/{out_name}",
            'filename': out_name
        })
    except Exception as e:
        app.logger.error(f"图片处理失败: {str(e)}")
        return jsonify({'success': False, 'message': f'处理图片时出错: {str(e)}'})


@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """下载文件"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': '文件不存在'}), 404
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    # 初始化
    init_processed_files()
    load_processors()
    
    # 启动后台线程
    t = threading.Thread(target=_background_watch, daemon=True)
    t.start()
    
    # 运行服务器
    app.run(host='0.0.0.0', port=5401, debug=True)