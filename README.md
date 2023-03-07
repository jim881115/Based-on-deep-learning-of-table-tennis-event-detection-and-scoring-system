# 基於深度學習之桌球事件偵測計分系統
## 專題介紹
* 建立一個使用簡單硬體便可完成的桌球計分系統
* 透過深度學習完成計分系統
* 創造成本更低並取代現有的計分系統

## 相關軟體及硬體
* 硬體
    * Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
    * Nvidia GTX1080Ti
    * 1080p 60fps 攝影機
* 軟體
    * Ubuntu 18.04
    * Tensorflow 2.3.1
    * OpenCV 4.4.0
    * Python3

## 系統架構
系統架構圖:<br/>
![](https://i.imgur.com/SpbtqXd.png) <br/>
* Ball Detection <br/>
　　透過類 VGG-16 style 的 ConNet 進行 Feature extraction，接著透過 DconvNet 所產生的特徵圖(Features map)進一步產生熱度圖(Heatmap)
* Event Spotting <br/>
　　Event Spottoing 將會沿用 Ball Detection 所產生的特徵圖，以減少系統負擔且有著更加準確的事件判斷
* Scoring Algorithm <br/>
　　參考以下流程圖為判斷得分的演算法 <br/>
![](https://i.imgur.com/y0zVYHO.png)

## 系統成果
![](https://i.imgur.com/mBmOaM4.png)
1. 當前比分，紫色為大局比分，黃色為當局比分，並且分數會隨著影片的進行，透過計分演算分來改變
2. 球體事件相關資訊，第一行為下一次要發生甚麼狀態才能滿足比賽繼續的條件。第二行為前一次的狀態，而最後一行為某一方發球後球總共發生了幾次彈跳
3. 透過醒目色彩為使用者提供球體位置，且將會滯留數幀畫面，使能夠更加明確地查看球體軌跡，並且也會標記出球體彈跳的位置
4. 標示出此次發球為哪一方所進行的
5. 目前球體事件為何。可分為彈跳事件、越網事件以及空事件

## 成果展示
![demo](./demo.gif)
