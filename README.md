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


