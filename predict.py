from ultralytics import YOLO

WEIGHTS   = r"C:\Users\89613\Desktop\flash_yolov12\yolov12\runs\train\yolov12n_face2_disease_cls_newnewor_a2\weights\best.pt"
DATA_YAML = r"C:\Users\89613\Desktop\apple\pre.yaml"  # 包含 images/ 和 labels/ 的根目录
DEVICE    = "cuda"                    # 或 "cpu"
IOU       = 0.45                      # 这里使用 iou 而不是 iouv

def batch_predict_and_eval():
    model = YOLO(WEIGHTS)
    model.fuse()

    # （可选）先预测并保存结果
    model.predict(
        source=r"C:\Users\89613\Desktop\apple\test\images",
        device=DEVICE,
        conf=0.25,
        iou=IOU,
        save=True,
        save_txt=True,
        project="runs/predict",
        name="exp_folder",
        exist_ok=True
    )

    # 直接调用 val，注意参数名改为 iou
    results = model.val(
        data=DATA_YAML,
        iou=IOU,
        device=DEVICE
    )

    # 取第一个 Results 对象，读取 precision 和 recall
    metrics = results[0].metrics
    print(f"Precision (P): {metrics['precision']:.4f}")
    print(f"Recall    (R): {metrics['recall']:.4f}")

if __name__ == "__main__":
    batch_predict_and_eval()
