
---

## ✅ English Version `README.md`

```markdown
# Improved YOLOv12: Enhanced Object Detection with CoordAtt and FMA

## 📌 Project Overview

This repository presents an enhanced version of the classical YOLOv12 object detection model. We introduce two major improvements—**CoordAtt** (Coordinate Attention Mechanism) and **FMA** (Feature Merge and Attention module)—to significantly boost detection performance in challenging environments.

The proposed model is suitable for various object detection scenarios such as agricultural disease detection, license plate recognition, industrial defect inspection, and more. All experiments were conducted using our custom dataset, and results indicate improved robustness against occlusion, complex backgrounds, and adversarial noise.

---

## 🔧 Model Enhancements

- **CoordAtt**: Embeds channel and spatial attention to refine feature representation.
- **FMA Module**: Enhances cross-layer feature fusion to improve small object detection.
- **Combined Optimization**: Integrating both modules delivers consistent performance improvement over the baseline.

---

## 📂 Dataset Access

We provide the dataset used for training and evaluation at the link below:  
👉 [Click to access dataset]
(https://pan.baidu.com/s/1R4yCCy8wzooD_zWUW57rLw 提取码: 6666 )

The dataset includes training, validation, and test splits.

---

## 🚀 Getting Started

# Install dependencies
pip install -r requirements.txt

# Train the model
python test.py --config config.yaml
