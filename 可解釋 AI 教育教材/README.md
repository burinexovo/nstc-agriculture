# 可解釋 AI 教育教材

教育各子計畫學會如何使用 Grad-CAM 進行模型可視化，看看模型到底在注意圖片的哪個地方。準備三個 .ipynb 範例，分別對應分類、分割、偵測這三種影像任務，會示範在不同情況下怎麼套用 Grad-CAM，並解釋每種任務需要注意的地方。

雲端位置（含簡報、模型權重）：[link](https://drive.google.com/drive/folders/1RAVrzAtW_BvALNFsLnODKCZ6CLQyVcKw?usp=drive_link)

---

## 方法概覽

Grad-CAM（Gradient-weighted Class Activation Mapping）是一種廣泛使用的可解釋 AI 方法，透過計算目標類別對最後一層卷積特徵圖的梯度，生成視覺化熱力圖，突顯模型做出預測時所關注的影像區域。

本教材涵蓋三種常見影像任務：

| 任務 | 範例檔案 | 骨幹模型 |
|------|----------|----------|
| 影像分類 | `AI explainability for Classification.ipynb` | ResNet-101（ImageNet） |
| 物件偵測 | `AI explainability for Object Detection.ipynb` | YOLOv8 |
| 影像分割 | `AI explainability for Segmentation.ipynb` | U²-Net |

---

## 目錄結構

```
可解釋 AI 教育教材/
├── AI explainability for Classification.ipynb   # 分類任務 Grad-CAM 範例
├── AI explainability for Object Detection.ipynb  # 偵測任務 Grad-CAM 範例
├── AI explainability for Segmentation.ipynb      # 分割任務 Grad-CAM 範例
├── 20230523101646.jpg                            # 稻穗測試圖片
└── README.md
```

---

## 各範例說明

### 1. 影像分類（Classification）

以 **ResNet-101**（ImageNet 預訓練）為例：

1. 安裝 `grad-cam` 套件（`!pip install grad-cam`）
2. 載入測試圖片（貓、狗等 ImageNet 類別）
3. 使用模型推論，取得預測類別
4. 選定目標卷積層（`model.layer4[-1].conv3`）
5. 使用 GradCAM 計算熱力圖並疊加於原圖顯示

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.layer4[-1].conv3]
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
```

### 2. 物件偵測（Object Detection）

以 **YOLOv8** 為例：

- 針對偵測框內的目標物件產生 Grad-CAM 熱力圖
- 說明 Grad-CAM 套用於偵測模型時，輸出層的選取方式與分類任務的差異

### 3. 影像分割（Segmentation）

以 **U²-Net**（稻穗去背）為例：

- 載入訓練好的 U²-Net 模型與稻穗測試圖片
- 進行前景分割推論，輸出二值遮罩
- 示範如何在分割模型上套用 Grad-CAM，觀察模型關注的語義區域

---

## 環境需求

```bash
pip install grad-cam
pip install torch torchvision
pip install matplotlib numpy pillow opencv-python
```

若使用物件偵測範例，需額外安裝：

```bash
pip install ultralytics
```

---

## 關鍵套件

| 套件 | 說明 |
|------|------|
| [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) | 提供 GradCAM、GradCAM++、ScoreCAM 等多種 CAM 方法 |
| `torchvision` | 提供 ResNet-101 等預訓練分類模型 |
| `ultralytics` | 提供 YOLOv8 模型與推論介面 |

---

## 注意事項

- 不同任務的目標層選取方式不同，請參考各 notebook 內的說明
- 分割任務的 Grad-CAM 目標為分割輸出的特定通道，而非分類 logit
- 完整技術細節與範例可參考雲端簡報
