# %%
from ultralytics import SAM
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# %%
image_path = "./origin_image/input1.jpg"

img = Image.open(image_path)
plt.imshow(img)
plt.show()

# %%
# 載入模型
# model = SAM('sam_b.pt')
model = SAM('sam_l.pt')

# 顯示模型資訊
model.info()

# 推論
results = model(image_path, points=[550, 200]) # points=[x, y]

# %%
masks = results[0].masks.data
binary_mask = (masks[0].cpu().numpy() > 0.5).astype(np.uint8) * 255

print(binary_mask.shape)

# 閉運算
kernel = np.ones((5,5), np.uint8)
closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

# 填充孔洞
contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    cv2.drawContours(closed_mask, [contour], 0, 255, -1)

plt.imshow(closed_mask, cmap='gray')
plt.show()

# %%
Image.fromarray(closed_mask).save('./mask_image/dog1.png')
