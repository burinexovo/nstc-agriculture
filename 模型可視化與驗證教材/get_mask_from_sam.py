import matplotlib.pyplot as plt  # 繪圖套件
import numpy as np  # 數值運算套件
from PIL import Image  # 圖像處理套件
import cv2  # 影像處理套件
from ultralytics import SAM  # SAM模型


# 像素位置全局變數
x = None
y = None


def onclick(event):
    '''滑鼠點擊事件處理函數'''
    global x, y

    if event.xdata is None or event.ydata is None:
        return

    # 獲取整數座標位置
    x = int(round(event.xdata))
    y = int(round(event.ydata))

    # 確保座標在圖像範圍內
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        # 獲取像素值
        pixel_value = image[y, x]

        # 如果是彩色圖片
        if len(image.shape) == 3:
            print(f'位置: (x={x}, y={y}), RGB值: {pixel_value}')
        # 如果是灰度圖片
        else:
            print(f'位置: (x={x}, y={y}), 像素值: {pixel_value}')

        plt.close('all') # 關閉所有圖形
        plt.clf()  # 清除當前圖形
        plt.cla()  # 清除當前軸

'''取得滑鼠點擊的像素位置'''
# 讀取圖片
image_path = "./origin_image/dog1.jpg"
image = np.array(Image.open(image_path))

# 創建圖形
fig, ax = plt.subplots()
ax.imshow(image)

# 連接點擊事件
fig.canvas.mpl_connect('button_press_event', onclick)

plt.title('Get pixel value by clicking')
plt.show()


'''載入模型並推論'''
# 載入模型
model = SAM('sam_b.pt')  # 基礎模型，速度較快
# model = SAM('sam_l.pt') # 較大模型，精度較高

# 推論
results = model.predict(
    image_path, 
    points=[x, y], 
    show=False,
    verbose=False
)

masks = results[0].masks.data
binary_mask = (masks[0].cpu().numpy() > 0.5).astype(np.uint8) * 255

print(binary_mask.shape)

# 閉運算
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

# 填充孔洞
contours, _ = cv2.findContours(
    closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    cv2.drawContours(closed_mask, [contour], 0, 255, -1)

Image.fromarray(closed_mask).save('mask.png')