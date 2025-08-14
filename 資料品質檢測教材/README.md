# 資料品質檢測

教育各子計畫利用 AutoEncoder 進行資料集品質檢測。在此方法中，我們先以品質相對較佳的圖片資料作為訓練集，讓 AutoEncoder 學習「優質資料」的特徵與模式。
當新的資料集加入時，將圖片輸入已訓練好的 AutoEncoder 進行重建：
- 重建效果佳 → 圖片與優質資料特徵相符，判定為品質良好。
- 重建效果差 → 圖片與優質資料差異大，判定為品質不佳。

透過此方式，即可自動將資料集劃分為「乾淨」與「不乾淨」兩部分，達到高效的品質檢測與篩選。
（完整技術細節與範例可參考雲端簡報）

雲端位置（含簡報）：[link](https://drive.google.com/drive/folders/1aiJxOTNUHWt5y6faeAVKoVRFa2lV2--T?usp=sharing)
