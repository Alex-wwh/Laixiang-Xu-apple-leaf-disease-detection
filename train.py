from ultralytics import YOLO
from multiprocessing import freeze_support
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    """
    训练 YOLOv12 模型（针对植物叶片病害分类）
    """
    try:
        # 加载预训练模型（针对分类）
        model = YOLO(r"C:\Users\89613\Desktop\flash_yolov12\yolov12\ultralytics\cfg\models\v12\yolome.yaml")  # 更正为正确的分类配置文件
        logger.info("预训练模型加载成功。")

        # 数据集配置文件路径（确保是整个数据集的data.yaml）
        data_config = r"C:\Users\89613\Downloads\archive (1)\data.yaml"  # 更正为根目录的data.yaml

        # 设置训练超参数（针对分类）
        logger.info("开始训练模型...")
        model.train(
            data=data_config,
            epochs=100,  # 训练轮次
            imgsz=640,  # 图像大小
            batch=16,  # 批量大小
            optimizer='AdamW',  # 优化器
            lr0=0.001,  # 初始学习率
            lrf=0.1,  # 学习率衰减比例
            weight_decay=0.0005,  # 权重衰减
            warmup_epochs=5,  # 预热轮次
            warmup_momentum=0.8,  # 预热动量
            project='runs/train',  # 保存路径
            name='yolov12n_faccce_disease_cls_newnewor_a',  # 训练名称（更正为分类任务）
            freeze=[10],  # 冻结前10层
            device='0'  # GPU设备
        )
        logger.info("模型训练完成。")

        # 验证模型性能
        logger.info("开始验证模型...")
        results = model.val()
        logger.info(f"验证准确率 top1: {results.top1:.3f}, top5: {results.top5:.3f}")

        # 导出模型为 ONNX 格式
        export_model(model, format='onnx')

    except Exception as e:
        logger.error(f"训练或验证过程中发生错误: {e}")

def export_model(model, format='onnx'):
    """导出模型"""
    try:
        logger.info(f"开始导出模型为 {format.upper()} 格式...")
        model.export(format=format, dynamic=True)
        logger.info(f"模型已导出为 {format.upper()} 格式")
    except Exception as e:
        logger.error(f"导出模型时发生错误: {e}")

def visualize_results():
    """可视化训练结果"""
    results_dir = 'runs/train/yolov12n_plant_disease_cls'
    if os.path.exists(results_dir):
        logger.info(f"训练结果已保存至: {results_dir}")
        # 可添加可视化代码
    else:
        logger.warning("未找到训练结果，请检查路径是否正确。")

if __name__ == '__main__':
    freeze_support()
    train_model()
    visualize_results()