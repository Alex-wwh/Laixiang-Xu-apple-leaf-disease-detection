import os
from ultralytics import YOLO
import cv2 # 可以用cv2来读取和处理图片，但ultralytics的predict方法通常直接接受路径并处理保存

def predict_and_save_image(model_path, image_path, output_dir="runs/predict", save_name=None):
    """
    加载YOLO模型，对指定图片进行预测，并将结果图片保存到目录。

    Args:
        model_path (str): 训练好的YOLO模型文件（.pt文件）的路径。
        image_path (str): 需要预测的图片文件的路径。
        output_dir (str): 保存预测结果图片的根目录。默认为 "runs/predict"。
                        结果图片会保存在 output_dir/predictN/ 这样的子文件夹中，N是自动递增的序号。
        save_name (str, optional): 可选。如果指定，结果图片会保存到 output_dir/predictN/save_name。
                                  如果为None，ultralytics会使用原始图片名。
    Returns:
        str or None: 保存的结果图片的完整路径（如果预测成功），否则返回None。
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件未找到：{model_path}")
        return None

    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片文件未找到：{image_path}")
        return None

    print(f"加载模型：{model_path}")
    # 加载模型
    model = YOLO(model_path)

    print(f"正在预测图片：{image_path}")

    # 执行预测
    # save=True 会自动保存预测结果图片
    # project 指定保存结果的根目录
    # name='' 可以让每次运行的结果都保存在 'output_dir/predict' 下，而不是创建 'predict1', 'predict2' 等
    # 但是使用 name='' 可能会覆盖之前的结果，更常用的方式是让它自动创建 predictN
    # 这里为了简单演示保存路径，我们假设ultralytics会保存到 output_dir 下的某个子文件夹
    results = model(image_path, save=True, project=output_dir)

    # 获取保存的图片路径
    # results 是一个列表，因为你可以同时对多个源进行预测。这里我们只预测一张图片，所以取第一个结果。
    saved_image_path = None
    if results and results[0].save_dir:
         # results[0].save_dir 是保存结果所在的目录 (例如 output_dir/predict)
         # 实际保存的文件名通常是原始文件名，或者如果你指定了 save_name
         saved_image_dir = results[0].save_dir
         if save_name:
             saved_image_path = os.path.join(saved_image_dir, save_name)
             # 注意：ultralytics的save=True会根据原始文件名自动命名，
             # 如果需要自定义文件名，可能需要更复杂的处理或在预测后手动重命名/复制
             # 简单起见，这里假设保存的文件名就是原始文件名或指定的save_name
             # 实际保存的文件名可以通过检查 saved_image_dir 目录内容来确认
             print(f"请检查目录 {saved_image_dir} 寻找保存的图片。")
             # 尝试构造可能的路径，但这不保证完全准确，以实际保存的文件名为准
             original_image_name = os.path.basename(image_path)
             possible_saved_path = os.path.join(saved_image_dir, save_name if save_name else original_image_name)
             print(f"可能的保存路径：{possible_saved_path}")
         else:
              original_image_name = os.path.basename(image_path)
              saved_image_path = os.path.join(saved_image_dir, original_image_name)
              print(f"预测结果图片已保存到：{saved_image_path}")


    return saved_image_path

# --- 示例用法 ---
if __name__ == "__main__":
    # TODO: 请替换为你的模型文件和图片文件的实际路径
    your_model_file_path = r"C:\Users\89613\Desktop\flash_yolov12\yolov12\runs\train\yolov12n_face2_disease_cls_newnewor_a2\weights\best.pt" # 例如: "runs/train/exp/weights/best.pt"
    your_image_file_path = r"C:\Users\89613\Desktop\io\微信图片_20250521084113.jpg"# 例如: "data/images/test/image1.jpg"

    # TODO: 指定保存结果的目录
    output_directory_for_results = r"C:\Users\89613\Desktop\io"

    print("--- 运行YOLO预测脚本 ---")

    # 创建输出目录（如果不存在）
    # 注意：ultralytics的save=True会自动创建output_dir下的子文件夹，
    # 但创建根目录是个好习惯
    os.makedirs(output_directory_for_results, exist_ok=True)

    # 执行预测并保存
    # 你可以选择指定一个保存的文件名 (save_name="predicted_result.jpg")，
    # 或者让它使用原始文件名 (save_name=None)
    saved_location = predict_and_save_image(
        model_path=your_model_file_path,
        image_path=your_image_file_path,
        output_dir=output_directory_for_results,
        save_name=None # 或指定一个文件名，例如 "my_predicted_image.jpg"
    )

    if saved_location:
        print(f"\n预测及保存完成。请查看路径：{saved_location} (注意：此路径是ultralytics构造的，以实际保存文件为准)")
    else:
        print("\n预测或保存失败。请检查错误信息。")