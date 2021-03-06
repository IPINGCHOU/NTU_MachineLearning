# NTU_MachineLearning
108-2 Machine Learning (機器學習) <br/>
由李宏毅教授、吳沛遠教授和林宗男教授共同合授。李宏毅教授負責授課與作業規劃。

## HW1 Linear Regression
  資料 : 行政院環境環保署空氣品質監測網所下載的觀測資料 <br/>
  目的 : 本作業實作 Linear Regression 預測出 PM2.5 的數值 <br/>
  實作 Linear Regression 並手刻 Adagrad 以及 Adam optimizer實現梯度下降。 <br/>
   + Simple baseline √  (By Linear Regression)<br/>
   + Strong baseline √  (By Linear Regression)

## HW2 Binary Classification
  資料 : Census-Income (KDD) Dataset <br/>
  目的 : whether the income of an individual exceeds $50000 or not? <br/>
  實作 Logistic Regression 以及 Probabilistic Generative model，其中 Logistic Regression 以 Cross-Entropy 作為 loss function 並實現梯度下降<br/>
   + Simple baseline √ (Both Logistic Regression and Probabilistic Generative model) <br/>
   + Strong baseline √ (By a simple NN model)

## HW3 CNN - Food Classification
資料：此次資料集為網路上蒐集到的食物照片，共有11類  <br/>
Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.  <br/>
+ Training set: 9866張 
+ Validation set: 3430張
+ Testing set: 3347張 <br/>
目的：食物分類 <br/>
實作 CNN 模型
   + Simple baseline √ (By a simple CNN model) <br/>
   + Strong baseline √ (By a classic VGG-16 model)
   
## HW4 Recurrent Neural Networks - Text Sentiment Classification
資料：為 Twitter 上收集到的推文，每則推文都會被標注為正面或負面
除了 labeled data 以外，還額外提供了 120 萬筆左右的 unlabeled data
+ labeled training data    ：20萬
+ unlabeled training data  ：120萬
+ testing data             ：20萬（10 萬 public，10 萬 private） <br/>
目的：語句分類 <br/>
實作 RNN 模型，使用LSTM
   + Simple baseline √ (By a simple LSTM model, with labeled training data) <br/>
   + Strong baseline √ (By a bidirectional LSTM deep model with more fc layers, fine tuned, with both label and unlabled data, over 90w sentences.) <br/>
   
## HW5 Explainable AI - on CNN model (HW3)
 資料：hw3 CNN model output <br/>  
 目的：視覺化 CNN model output 
  + Task1 - Sailency Map
  + Task2 - Filter Visualization
  + Task3 - Lime
  + Task4 - Any visualization/explaining method you like - (SHAP)
  
## HW6 Adversarial attack - on black box CNN model 
資料：food-11 dataset <br/>
目的：attack black box model by FGSM-attack or any other attack for better result <br/>
Proxy model: (pytorch-pretrained)
  + VGG-16
  + VGG-19
  + ResNet-50
  + ResNet-101
  + DenseNet-121
  + DenseNet-169  <br/> 
實作：以 FGSM 通過 simple baseline，並以任意方法通過 strong baseline
  + Simple baseline √ (By FGSM attack, eps = 0.3, acc = 0.07, L-inf = 16.7250) <br/>
  + Strong baseline √ (By basic iterative method FGSM (I-FGSM), eps = 0.03, alpha = 0.005, acc = 0, L-inf = 8.4000) <br/>
  + Also tried One-Pixel-Attack, but with slow computation speed and bad result :(

## HW7 Network Compression 
  pass
## HW8 Seq2seq - en 2 cn
資料：manythings之cmn-eng
  + train : 18000 sents
  + valid : 500   sents
  + test  : 2636  sents  <br/>
目的：英翻中，每次輸入單一句子  <br/>
實作：利用 GRU 建立 Seq2Seq 模型，分別實現以下 task 並比較。
  + Teacher Forcing
  + Attention Mechanism
  + Beam Search
  + Schedule Sampling   <br/>
此次並無baseline，Beam Search以heapq實現，Schedule sampling選擇使用 Linear, Exponential, Inverse Sigmoid實現。可參見 : https://arxiv.org/abs/1506.03099
 
## HW9 Unsupervised Learning
資料：?
  + train : 8500pics, 32*32*3
  + valX : 500pics, 32*32*3
  + valY : labels, 500 <br/>
目的 : 將照片分為風景與非風景照 <br/>
實作 : 建立 autoencoder，並對其降維後分群 <br/>
  + Simple baseline √ (By tutorial, simple CNN layers, acc = 0.74918) <br/>
  + Strong baseline √ (By deeper CNN modle and bm for each layer, acc = 0.78941) <br/>
  + U-Net, simple U-Net with bm, better reconstruction but lower acc :(  (acc = 0.73789)
  
## HW10 Anomaly Detection

## HW11 GAN
資料：faces by Crypko, https://crypko.ai/#/
+ train : 10000pics, 64*64*3 <br/>
目的：訓練出GAN
實作：DCGAN, WGAN
此次作業無Kaggle，實作DCGAN觀察mode collapse以及實作WGAN避免mode collapse。詳細列於report.pdf之中

## HW12 Transfer Learning
資料：?
  + train : 5000 pics, 32*32*3, 10 labels, real RGB images
  + target : 100,000 pics, 28*28*1, no labels, graffiti images <br/>
目的：只訓練在train label上，在不需要target label的情況下提升target label的預測準確度。
實作：DaNN, MSDA
  + Simple baseline √ (By tutorial DaNN, Source : Canny transfer, acc = 0.57270) <br/>
  + Strong baseline √ (By MSDA, Source : Canny, Sobel, Laplacian, Gray transfer, acc = 0.75790) <br/>
MSDA reference : https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit

## HW13 Meta Learning
  pass
## HW14 Life-long Learning
  pass
## HW15 Reinforcement Learning
資料 : gym - Lunar Lander <br/>
目的：以PG based algorithm讓登陸艇成功登陸月球~ <br/>
實作：<br/>
+ PG
+ discount reward with PG
+ A2C
+ discount reward with n step info A2C


