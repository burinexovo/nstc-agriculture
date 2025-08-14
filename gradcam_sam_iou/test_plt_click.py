import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def onclick(event):
    """滑鼠點擊事件處理函數"""
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

# 讀取圖片
image = np.array(Image.open('origin_image/elephant1.jpg'))

# 創建圖形
fig, ax = plt.subplots()
ax.imshow(image)

# 連接點擊事件
fig.canvas.mpl_connect('button_press_event', onclick)

plt.title('Get pixel value by clicking')
plt.show()