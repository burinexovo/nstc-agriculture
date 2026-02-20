# 資料品質檢測教材

教育各子計畫利用 AutoEncoder 進行資料集品質檢測。在此方法中，我們先以品質相對較佳的圖片資料作為訓練集，讓 AutoEncoder 學習「優質資料」的特徵與模式。當新的資料集加入時，將圖片輸入已訓練好的 AutoEncoder 進行重建：

- 重建效果佳 → 圖片與優質資料特徵相符，判定為品質良好。
- 重建效果差 → 圖片與優質資料差異大，判定為品質不佳。

透過此方式，即可自動將資料集劃分為「乾淨」與「不乾淨」兩部分，達到高效的品質檢測與篩選。

雲端位置（含簡報）：[link](https://drive.google.com/drive/folders/1aiJxOTNUHWt5y6faeAVKoVRFa2lV2--T?usp=sharing)

---

## 方法概覽

![影像資料品質檢測方法（OpenCV + AutoEncoder）](../assets/photos/影像資料品質檢測方法（OpenCV%20+%20AutoEncoder）.jpg)

完整流程分為兩階段：

**階段一（前處理）：OpenCV 快速過濾**
- 使用拉普拉斯算子偵測模糊圖片（清晰度分數低於閾值者排除）
- 分析像素亮度直方圖，過濾過曝與欠曝圖片

**階段二（深度學習）：AutoEncoder 異常偵測**
- 以過濾後的正常圖像訓練卷積式 AutoEncoder
- 對新進圖像計算重建誤差（MSE 與 SSIM 雙指標）
- 比較重建誤差與閾值，自動判定品質是否合格

---

## 目錄結構

```
資料品質檢測教材/
├── Data Quality Assessment_MNIST.ipynb              # MNIST 入門教學範例
├── Data Quality Assessment_苗株可見光影像資料集.ipynb  # 農業影像實作範例
├── image/                                            # 苗株影像資料集（執行後自動下載）
│   ├── raw_data/                                     # 原始正常影像（100 張）
│   └── destroy_data/                                 # 損毀異常影像（6 張）
└── README.md
```

---

## 範例一：MNIST 入門教學（`Data Quality Assessment_MNIST.ipynb`）

以 MNIST 手寫數字資料集作為入門範例，示範 AutoEncoder 品質檢測的核心原理。

### 資料定義

| 類型 | 內容 | 數量 |
|------|------|------|
| 正常資料 | 手寫數字 '1' | 6,742 張（前 2,000 張用於訓練） |
| 異常資料 | 數字 '0' 與 '2'～'9' | 53,258 張 |

### 執行步驟

| 步驟 | 說明 |
|------|------|
| 1. 載入資料集 | 使用 TensorFlow API 載入 MNIST（6 萬張訓練圖片） |
| 2. 定義正常/異常 | 數字 '1' 為正常，其餘為異常 |
| 3. 訓練 AutoEncoder | 以 2,000 張 '1' 的圖像訓練，共 30 個 epoch |
| 4. 設定閾值 | 以訓練集重建誤差的「平均值 + 3×標準差」為閾值 |
| 5. 異常偵測 | 對測試集推論，超過 MSE 或 SSIM 閾值者判定為異常 |

### 模型架構

卷積式 AutoEncoder（無 Pooling 層，保留完整空間資訊）：

```
Encoder: Conv2D(32) → Conv2D(64) → Conv2D(128)
Decoder: Conv2D(128) → Conv2D(64) → Conv2D(32) → Conv2D(1, sigmoid)
```

- 輸入/輸出尺寸：28×28×1
- 總參數量：332,801
- 損失函數：MSE

### 範例結果

| 資料集 | 正常（判定） | 異常（判定） |
|--------|-------------|-------------|
| 正常測試集（100 張 '1'） | 98 | 2 |
| 異常資料（100 張非 '1'） | 少數 | 大部分正確識別 |

---

## 範例二：苗株可見光影像實作（`Data Quality Assessment_苗株可見光影像資料集.ipynb`）

以農業苗株可見光影像資料集為例，展示完整的兩階段品質檢測流程。

### 資料集說明

| 類型 | 說明 | 數量 |
|------|------|------|
| 正常影像（raw_data） | 苗株可見光正常圖像 | 100 張（80 訓練 / 20 測試） |
| 損毀影像（destroy_data） | 人工損毀的異常圖像 | 6 張 |

使用 `gdown` 從 Google Drive 自動下載至 `image/` 目錄。

### 執行步驟

| 步驟 | 說明 |
|------|------|
| 1. 載入資料集 | 使用 `gdown` 下載，`load_images()` 讀取並 resize 至 128×128 |
| 2. 定義正常/異常 | 80 張正常圖像為訓練集，20 張為測試集，6 張損毀圖像為異常集 |
| 3. OpenCV 過濾 | 拉普拉斯模糊偵測（mean−2σ 閾值）＋亮度直方圖過曝/欠曝偵測 |
| 4. 訓練 AutoEncoder | 以過濾後的正常圖像訓練，共 30 個 epoch |
| 5. 設定閾值 | 以訓練集重建誤差的「平均值 + 3×標準差」為閾值 |
| 6. 異常偵測 | MSE 與 SSIM 雙指標判定，任一超標即視為異常 |

### OpenCV 過濾細節

**模糊偵測（拉普拉斯算子）**

```python
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
blur_score = np.mean(np.abs(laplacian))
# 閾值 = mean - 2 * std，低於閾值者排除
```

**過曝/欠曝偵測（亮度直方圖）**

```python
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
upper_threshold = np.percentile(hist, 90)  # 過曝閾值
lower_threshold = np.percentile(hist, 10)  # 欠曝閾值
```

### 模型架構

卷積式 AutoEncoder（彩色影像版本）：

```
Encoder: Conv2D(32) → Conv2D(64) → Conv2D(128)
Decoder: Conv2D(128) → Conv2D(64) → Conv2D(32) → Conv2D(3, sigmoid)
```

- 輸入/輸出尺寸：128×128×3
- 總參數量：333,955
- 損失函數：MSE，評估指標另含 SSIM Loss

### 範例結果

| 資料集 | 正常（判定） | 異常（判定） |
|--------|-------------|-------------|
| 測試集（20 張正常） | 20 | 0 |
| 損毀集（6 張異常） | 1 | 5 |

---

## 環境需求

```bash
pip install tensorflow
pip install numpy matplotlib opencv-python gdown
```

---

## 閾值設定說明

```python
k = 3  # 可根據資料集特性調整
threshold = mean_error + k * std_error
```

| k 值 | 效果 |
|------|------|
| 較大（如 k=3） | 閾值寬鬆，誤報率低，但可能漏報部分異常 |
| 較小（如 k=1） | 閾值嚴格，偵測敏感，但誤報率上升 |

建議以驗證集調整 k 值，以達到最佳的精確率/召回率平衡。

---

## 注意事項

- 苗株影像資料集需透過 `gdown` 從 Google Drive 下載，請確保網路連線穩定，且已安裝 `gdown` 套件
- AutoEncoder 訓練效果高度依賴訓練集的品質與多樣性，訓練集圖像數量過少時建議搭配資料增強
- 雙指標（MSE + SSIM）任一超標即判定為異常，若誤報率偏高可考慮改為需兩者同時超標
- 完整技術細節與範例可參考雲端簡報
