import cv2
import os
from ultralytics import YOLO

def predict_images(model_path, image_paths, save_dir='runs/detect/predict5'):
    """
    批量预测图片并保存结果
    :param model_path: 模型路径(.pt文件)
    :param image_paths: 图片路径列表
    :param save_dir: 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    try:
        # 加载训练好的模型
        model = YOLO(model_path)
        print(f"成功加载模型：{model_path}")

        # 批量推理
        results = model.predict(
            source=image_paths,
            save=True,  # 自动保存带标注的结果
            save_dir=save_dir,  # 指定保存目录
            show_labels=True,  # 显示标签
            show_conf=True,  # 显示置信度
            conf=0.5,  # 置信度阈值
            line_width=1  # 框线粗细
        )

        print(f"\n预测完成！结果已保存至：{os.path.abspath(save_dir)}")

        # 保存检测结果
        for i, (r, img_path) in enumerate(zip(results, image_paths)):
            img_with_boxes = r.plot()  # 获取绘制了检测框的图像
            save_path = os.path.join(save_dir, os.path.basename(img_path))  # 生成保存路径
            cv2.imwrite(save_path, img_with_boxes)  # 保存图像
            print(f"已保存：{save_path}")

        print(f"\n所有预测图片已保存至：{os.path.abspath(save_dir)}")

    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    # 要预测的图片路径列表（修改为你的实际路径）
    test_images = [
        r"C:\Users\89613\Desktop\检测\GLP-OEB0322_jpg.rf.eaccfa7faece4998f34ae6819eb7d48a.jpg",
        r"C:\Users\89613\Desktop\检测\IMG_20250228_175852_jpg.rf.e6be50fcff4e0a9603d576540d1e6bf3.jpg",
        r"C:\Users\89613\Desktop\检测\Val-blue-card222_jpg.rf.becb0faa43df45d38980826fc2a604ad.jpg",
        r"C:\Users\89613\Desktop\检测\YLP-OEB0292_jpg.rf.312e1d4bfd0d42b842212065090af163.jpg"
    ]

    # 训练好的模型路径（修改为你的实际路径）
    model_path = r"C:\Users\89613\Desktop\flash_yolov12\yolov12\runs\train\yolov12n_improve_fma+coord5\weights\best.pt"

    # 执行预测
    predict_images(model_path, test_images)
