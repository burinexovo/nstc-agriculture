# Explainable AI for Smart Sustainable Agriculture

> 國科會計畫「智慧永續新農業研究發展中心」— 可解釋人工智慧（XAI）研究與教育模組

本專案為國科會（NSTC）智慧農業計畫之研究成果，聚焦於深度學習模型的**可解釋性（Explainability）**與**可信度驗證**，應用於稻穗偵測、茶葉病蟲害辨識、苗株表型分析等農業場景。專案涵蓋研究方法開發與子計畫教育教材設計兩大面向。

## Highlights

- **模型可視化驗證流程**：結合 Grad-CAM / Score-CAM 與 SAM（Segment Anything Model），以 IoU 量化模型關注區域與人類認知的一致性
- **多輪超像素遮蔽敏感度量化方法**：提出基於 SLIC 超像素分割的多輪隨機遮蔽實驗，以 Dice Similarity Score 與 Masking Impact Score 量化模型對區域的依賴度與穩定性
- **Score-CAM 替代方案**：引入不依賴梯度的 Score-CAM，解決 Grad-CAM 梯度消失與噪聲干擾問題
- **資料品質自動檢測**：以 AutoEncoder 重建誤差進行異常資料偵測，確保訓練集品質
- **教育教材模組**：為各子計畫設計一系列 XAI 教學 Notebook，涵蓋分類、分割、物件偵測三大任務

## Project Structure

```
.
├── 多輪超像素遮蔽敏感度量化方法/     # 核心研究：多輪遮蔽敏感度量化
│   ├── model/
│   │   └── u2net.py                  #   U²-Net 語意分割模型架構
│   ├── modules/
│   │   └── dataset.py                #   資料載入與前處理 Pipeline
│   ├── extract.ipynb                 #   稻穗分割推論流程
│   ├── evaluate.ipynb                #   DSC / MIS 指標計算與分析
│   └── LIME_extract.ipynb            #   LIME 替代 XAI 方法比較
│
├── 模型可視化與驗證教材/              # Grad-CAM + SAM IoU 驗證教材
│   ├── vgg16_gradcam_iou_for_torch.ipynb
│   └── get_mask_from_sam.py
│
├── 可解釋 AI 教育教材/               # XAI 基礎教育（分類/分割/偵測）
│   ├── AI explainability for Classification.ipynb
│   ├── AI explainability for Segmentation.ipynb
│   └── AI explainability for Object Detection.ipynb
│
├── 資料品質檢測教材/                  # AutoEncoder 資料品質檢測
│   ├── Data Quality Assessment_MNIST.ipynb
│   └── Data Quality Assessment_苗株可見光影像資料集.ipynb
│
├── scorecam/                          # Score-CAM 實作
│   ├── cam/
│   │   ├── basecam.py                #   Hook-based CAM 基底類別
│   │   └── scorecam.py               #   Score-CAM 演算法
│   └── vgg16_scorecam.ipynb
│
├── gradcam_sam_iou/                   # Grad-CAM + SAM 標準範例
├── gradcam_sam_iou_sub4/              # 子計畫4（茶葉病蟲害）範例
└── 10月期末報告撰寫方向.md
```

## Research Methods

### 1. 模型可視化驗證（Grad-CAM / Score-CAM + SAM + IoU）

回應國科會「如何確認模型可視化結果與人類認知一致」之問題，設計量化驗證流程：

```
輸入影像 → Grad-CAM / Score-CAM 產生熱力圖
                                        ↘
                                    IoU 計算 → 量化一致性分數
                                        ↗
輸入影像 → SAM 產生人工標註遮罩
```

- 以多個二值化閾值（0.05 間隔）將熱力圖轉換為 Mask
- 與 SAM 互動式分割結果計算 IoU（Intersection over Union）
- 依閾值範圍分組統計，提供可信度量化依據

### 2. 多輪超像素遮蔽敏感度量化方法

針對語意分割模型（U²-Net），提出多輪隨機遮蔽實驗以量化模型行為：

1. **SLIC 超像素分割**：將影像分割為具語義關聯性的區塊
2. **多輪隨機遮蔽**：每輪隨機遮蔽不同數量的超像素（填充黑色），重複 100+ 輪
3. **頻率統計**：統計各像素在多輪推論中被預測為目標的頻率
4. **高頻區域提取**：以閾值（如 >60%）提取穩定預測區域

**評估指標：**

| 指標 | 公式 | 意義 |
|------|------|------|
| **Dice Similarity Score (DSC)** | `2\|A∩B\| / (\|A\|+\|B\|)` | 高頻遮罩與單次輸出的一致性，高 DSC 表示模型預測穩定 |
| **Masking Impact Score (MIS)** | `1 - Σ IoU(原圖, 遮蔽圖) / N` | 遮蔽對模型輸出的影響程度，高 MIS 表示模型對該區域高度依賴 |

### 3. Score-CAM

引入 [Score-CAM（CVPR 2020 Workshop）](https://arxiv.org/abs/1910.01279) 作為 Grad-CAM 的替代方案：

- 不依賴梯度反向傳播，避免梯度飽和與不穩定問題
- 以各通道 Activation Map 對輸入影像加權遮罩，測量預測分數變化作為通道權重
- 實驗證明在關鍵區域定位上更為準確穩定

### 4. 資料品質檢測

以 AutoEncoder 重建誤差為基礎的異常資料偵測：

- 以乾淨資料訓練 Convolutional AutoEncoder
- 計算重建誤差（MSE / SSIM）
- 閾值設定：`μ + 3σ`，超過閾值判定為汙染資料
- 自動分離資料集為「乾淨」與「汙染」子集

## Tech Stack

| 類別 | 技術 |
|------|------|
| Deep Learning | PyTorch, TensorFlow / Keras |
| Models | U²-Net, VGG16, ResNet101, SAM (Base / Large) |
| XAI Methods | Grad-CAM, Score-CAM, LIME, Occlusion Sensitivity |
| Image Processing | OpenCV, scikit-image (SLIC), Pillow |
| Data Science | NumPy, SciPy, Matplotlib |
| Metrics | IoU, Dice Score, SSIM, MSE |

## Application Domains

| 應用場域 | 任務 | 對應模組 |
|----------|------|----------|
| 稻穗偵測 | 語意分割（U²-Net） | 多輪超像素遮蔽敏感度量化方法 |
| 茶葉病蟲害辨識 | 分類模型驗證 | gradcam_sam_iou_sub4 |
| 苗株表型分析 | 資料品質把關 | 資料品質檢測教材 |

## Getting Started

### Prerequisites

```bash
pip install torch torchvision tensorflow opencv-python scikit-image matplotlib pillow numpy scipy
pip install pytorch-grad-cam ultralytics
```

### Quick Start

1. **XAI 教學**：開啟 `可解釋 AI 教育教材/` 下的 Notebook，了解 Grad-CAM 在分類、分割、偵測任務上的應用
2. **模型驗證**：開啟 `模型可視化與驗證教材/vgg16_gradcam_iou_for_torch.ipynb`，實作 Grad-CAM + SAM IoU 驗證流程
3. **遮蔽量化**：依序執行 `多輪超像素遮蔽敏感度量化方法/` 下的 `extract.ipynb` → `evaluate.ipynb`
4. **資料品質**：開啟 `資料品質檢測教材/` 下的 Notebook，學習 AutoEncoder 異常偵測

> **Note:** 模型權重檔與完整簡報資料存放於 Google Drive，請參照各目錄內的說明連結。

## Acknowledgments

本研究為**國科會「智慧永續新農業研究發展中心」計畫**之成果，感謝計畫團隊與各子計畫的協作與支持。
