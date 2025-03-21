import os
import logging
import time
from datetime import datetime

def setup_logger(log_dir):
    """
    设置日志记录器
    
    Args:
        log_dir: 日志目录
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志记录器
    logger = logging.getLogger("rl_robot_control")
    logger.setLevel(logging.INFO)
    
    # 清除之前的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录初始信息
    logger.info(f"日志保存路径: {log_file}")
    logger.info(f"开始时间: {timestamp}")
    
    return logger


class MetricLogger:
    """
    指标记录器，用于记录训练过程中的各种指标
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def log(self, key, value, step):
        """
        记录指标
        
        Args:
            key: 指标名称
            value: 指标值
            step: 步数
        """
        if key not in self.metrics:
            self.metrics[key] = []
        
        self.metrics[key].append((step, value))
    
    def get_metrics(self, key=None):
        """
        获取指标数据
        
        Args:
            key: 指标名称，如果为None则返回所有指标
            
        Returns:
            指标数据
        """
        if key is not None:
            return self.metrics.get(key, [])
        return self.metrics
    
    def get_elapsed_time(self):
        """
        获取经过的时间
        
        Returns:
            经过的时间（秒）
        """
        return time.time() - self.start_time 