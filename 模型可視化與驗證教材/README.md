# 模型可視化與驗證：Grad-CAM 與 SAM 結合實踐

本範例展示了如何結合 **Grad-CAM 熱力圖** 與 **SAM 分割模型** 來進行模型可視化與驗證。通過計算 Grad-CAM 熱力圖與 SAM 遮罩的 **IoU (Intersection over Union)**，評估模型關注區域的準確性。

---

## 目錄結構說明

```palntext
origin_image/        # 測試圖片存放資料夾
mask_image/          # 測試圖片對應的遮罩存放資料夾
result/              # IoU 驗證結果輸出資料夾

imagenet_classes.txt # ImageNet 類別標籤檔案
get_mask_from_sam.py # 使用 SAM 模型生成遮罩的程式
sam_b.pt             # SAM 模型的基礎權重檔，執行速度較快
sam_l.pt             # SAM 模型的大型權重檔，分割精度較高
requirements.txt     # 此專案所需的 Python 套件列表
vgg16_gradcam_iou_for_torch.ipynb  # 主程式範例，計算 Grad-CAM 與 SAM 遮罩之 IoU
```

## 使用說明

### 1. 安裝環境

步驟 1: 安裝 PyTorch
請先確保已安裝 PyTorch，根據您的硬體配置（如 CUDA 支援）到 PyTorch 官方網站 選擇適合的版本並安裝。

步驟 2: 安裝專案依賴套件
安裝所需的 Python 套件列表：

```bash
pip install -r requirements.txt
```

### 2. 準備資料

測試圖片： 將測試圖像存放於 origin_image/ 資料夾中。

遮罩資料： 若測試圖片尚無遮罩，可使用 get_mask_from_sam.py 生成對應的遮罩，指令如下：

```bash
python get_mask_from_sam.py
```

### 3. 執行 IoU 計算

使用 vgg16_gradcam_iou_for_torch.ipynb，通過以下步驟完成 IoU 計算：

1. 將 origin_image/ 中的測試圖片與 mask_image/ 中的遮罩配對。
2. 依次計算 Grad-CAM 熱力圖與遮罩的 IoU。
3. 將結果按照 IoU 閾值（每 0.05 為一區間）存放於 result/ 資料夾。
   > 注意：若需要使用 SAM 不同精度的模型，可根據需求選擇 sam_b.pt 或 sam_l.pt。

## 檔案詳細說明

### 資料夾

- `origin_image/`
  測試圖片存放位置，必須包含要進行分析的圖像。

- `mask_image/`
  測試圖片對應的遮罩，若尚無遮罩，可使用 `get_mask_from_sam.py` 生成。

- `result/`
  存放 IoU 計算的最終結果。

### 主要檔案

- `imagenet_classes.txt`
  包含 ImageNet 類別標籤，用於分類模型的輸出解釋。

- `get_mask_from_sam.py`
  使用 SAM (Segment Anything Model) 自動生成圖像的遮罩。

  **使用範例：**

```bash
python get_mask_from_sam.py
```

- `sam_b.pt` 和 `sam_l.pt`
  - `sam_b.pt`：基礎權重檔，運行速度快，適合較小模型需求。
  - `sam_l.pt`：大型權重檔，分割精度較高，適合高精度場景。
- `requirements.txt`
  包含專案所需的套件，例如 pytorch, opencv-python, pytorch-grad-cam, ultralytics 等。

- `vgg16_gradcam_iou_for_torch.ipynb`
  主程式，示範如何使用 VGG16 分類模型生成 Grad-CAM 熱力圖，並計算與 SAM 遮罩的 IoU。 功能：

  - 自動加載 origin_image/ 和 mask_image/ 中的數據。
  - 設定 IoU 閾值範圍（0.05 間隔）。
  - 將結果輸出至 result/ 資料夾。

## 結果輸出說明

執行完成後，以下是結果存放位置及格式：

- 輸出資料夾：result/
- 結果內容：
  - IoU 計算結果以資料夾存儲，記錄不同閾值下的 IoU(0.05 為一個區間)。
