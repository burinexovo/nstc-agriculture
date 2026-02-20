# 模型可視化與驗證教材

先前國科會曾提出疑問：「如何確認 Grad-CAM 可視化的關注區域，與人類認知的關鍵區域一致？」為此，我們設計了將 Grad-CAM 與 SAM（Segment Anything Model）結合的方法。SAM 具備精確分割影像中多個物體的能力，使用者可先透過 SAM 選取並切割出自己關注的高精度區域，再與 Grad-CAM 的可視化結果進行比對，由此評估模型關注區域與人類認知區域的相似程度。

雲端位置（含簡報、模型權重）：[link](https://drive.google.com/drive/folders/1HJnPh90TLFalab5fMr61XtOaFIIJRF61?usp=drive_link)

---

## 方法概覽

本教材的核心問題是：**模型到底在看哪裡？它關注的地方是我們期望的嗎？**

透過以下三步驟進行量化驗證：

1. **Grad-CAM**：從 VGG-16 最後一層卷積層（`features[28]`）提取熱力圖，以 0.5 為閾值二值化，得到「模型關注區域」
2. **SAM**：透過互動式點擊介面，讓使用者指定「人類期望的關鍵區域」，SAM 精確切割出對應遮罩
3. **IoU 比對**：計算兩份遮罩的交並比，量化模型關注與人類認知的一致程度，並依分數分類儲存

---

## 目錄結構

```
模型可視化與驗證教材/
├── vgg16_gradcam_iou_for_torch.ipynb  # 主流程：Grad-CAM 生成 + IoU 驗證
├── get_mask_from_sam.py               # 互動式 SAM 遮罩提取工具
├── imagenet_classes.txt               # ImageNet 1000 類別標籤
├── requirements.txt                   # 相依套件清單
├── origin_image/                      # 測試原始圖像
│   ├── cat1.jpg / cat2.jpg
│   ├── dog1.jpg / dog2.jpg
│   └── elephant1.jpg / elephant2.jpg
├── mask_image/                        # SAM 產生的人工標記遮罩
│   ├── cat1.png / cat2.png
│   ├── dog1.png / dog2.png
│   └── elephant1.png / elephant2.png
└── result/                            # 驗證結果（依 IoU 分數區間分資料夾儲存）
    ├── 0.15-0.20/
    └── 0.50-0.55/
```

---

## 執行步驟

### Step 1：以 SAM 取得人工標記遮罩（`get_mask_from_sam.py`）

執行後顯示互動式圖像視窗，點擊想要關注的物體後，SAM 自動分割並輸出遮罩：

```bash
python get_mask_from_sam.py
```

核心流程：
1. 開啟圖像，透過滑鼠點擊取得像素座標 `(x, y)`
2. 載入 SAM 模型（`sam_b.pt` 基礎版 或 `sam_l.pt` 大型版）
3. 以點擊座標作為 prompt，SAM 輸出對應的分割遮罩
4. 對遮罩進行形態學閉運算（kernel 5×5）與孔洞填充，輸出乾淨的二值遮罩
5. 儲存為 `mask.png`，複製至 `mask_image/` 目錄備用

```python
model = SAM('sam_b.pt')  # 基礎模型，速度較快
# model = SAM('sam_l.pt')  # 較大模型，精度較高

results = model.predict(image_path, points=[x, y], show=False, verbose=False)
binary_mask = (masks[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
```

### Step 2：Grad-CAM 生成 + IoU 驗證（`vgg16_gradcam_iou_for_torch.ipynb`）

| 步驟 | 說明 |
|------|------|
| 1. 載入模型 | 載入預訓練 VGG-16（ImageNet 權重），選定 `features[28]` 為目標卷積層 |
| 2. 圖像推論 | 將圖像 resize 至 224×224，輸入模型取得預測類別索引 |
| 3. Grad-CAM | 以預測類別對 `features[28]` 計算梯度，生成灰度熱力圖 |
| 4. 二值化 | 以 0.5 為閾值將熱力圖轉為二值遮罩（模型關注區域） |
| 5. 載入 SAM 遮罩 | 讀取對應的人工標記遮罩並正規化至 [0, 1] |
| 6. 計算 IoU | 計算 Grad-CAM 遮罩與 SAM 遮罩的交並比 |
| 7. 儲存結果 | 依 IoU 分數以 0.05 為區間建立子資料夾，儲存原始圖像 |

```python
from pytorch_grad_cam import GradCAM

target_layers = [model.features[28]]
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

# 二值化（閾值 0.5）
binary_cam = (grayscale_cam > 0.5).astype(np.uint8)

# 計算 IoU
iou_score = calculate_iou(real_binary_cam, binary_cam)
```

---

## IoU 計算公式

$$IoU = \frac{|A \cap B|}{|A \cup B|}$$

其中 $A$ 為 Grad-CAM 二值遮罩，$B$ 為 SAM 人工標記遮罩。

### IoU 分數解讀

| IoU 分數 | 解讀 |
|----------|------|
| > 0.7 | 模型關注區域與人類期望高度一致，可信度高 |
| 0.5 ～ 0.7 | 關注區域大致重疊，模型行為合理 |
| < 0.5 | 模型關注區域與人類期望有明顯落差，需進一步審視 |

> 範例結果（`elephant2.jpg`，預測類別：tusker）：IoU = **0.5089**（儲存至 `result/0.50-0.55/`）

### 結果自動分類

```python
# 以 0.05 為區間，自動建立資料夾並儲存圖像
folder_base = (iou_score // 0.05) * 0.05
folder = f'{folder_base:.2f}-{folder_base + 0.05:.2f}'
os.makedirs(f'./result/{folder}', exist_ok=True)
```

---

## 模型選擇

| 模型 | 用途 | 說明 |
|------|------|------|
| VGG-16 | Grad-CAM 可視化 | ImageNet 預訓練分類模型，目標層：`features[28]`（最後一層 Conv2d） |
| SAM `sam_b.pt` | 遮罩提取 | Meta AI Segment Anything 基礎版，速度快 |
| SAM `sam_l.pt` | 遮罩提取 | 大型版，精度較高但速度較慢 |

---

## 環境需求

安裝方式請參考 `requirements.txt`，或手動安裝：

```bash
# 請先依照 https://pytorch.org/get-started/locally/ 安裝 PyTorch
pip install grad-cam
pip install ultralytics       # 包含 SAM 支援，sam_b.pt 首次執行時自動下載
pip install numpy matplotlib pillow opencv-python
```

---

## 注意事項

- `get_mask_from_sam.py` 需在支援 GUI 顯示的環境執行（本地端，或具顯示功能的遠端環境）
- SAM 模型權重（`sam_b.pt`）會在首次執行時由 Ultralytics 自動下載
- Grad-CAM 二值化閾值（預設 0.5）可依需求調整，影響遮罩的覆蓋範圍與 IoU 結果
- 完整技術細節與範例可參考雲端簡報
