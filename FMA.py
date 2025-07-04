import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载原始图像
image = cv2.imread(r"C:\Users\89613\Desktop\BLP-FV0195_jpg.rf.207defa09a440e76fe9687a4b7dbc184.jpg")  # 替换为你的图像路径
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊以减少噪声
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 使用 Canny 边缘检测算法
edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

# 显示原始图像和 Canny 边缘图
plt.figure(figsize=(10, 5))

# 原始图像
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Canny 边缘图
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.show()

# 保存 Canny 边缘图像
cv2.imwrite('../../canny_edge_image.jpg', edges)  # 保存处理后的边缘图像
