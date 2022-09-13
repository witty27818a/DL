# Deep Learning
Deep Learning course at NYCU

## Lab 1
### 概念: 反向傳播(手刻)
**_Lab1-Backpropagation.pptx_**: spec  
**_Lab1_DL.ipynb_**: 程式碼  
**_Report.pdf_**: 報告

## Lab 2
### 概念: CNN(串串看)
**_DL_Lab2_demo.ipynb_**: demo用的程式碼  
**_Lab2_DL_DeepConvNet.ipynb_**: **DeepConvNet**架構的程式碼  
**_Lab2_DL_EEGNet.ipynb_**: **EEGNet**架構的程式碼  
**_Lab2_EEG_classifications.pdf_**: spec  
**_Report.pdf_**: 報告  
**_S4b_test.npz_**: 一部分測試資料  
**_S4b_train.npz_**: 一部份訓練資料  
**_X11b_test.npz_**: 另一部分測試資料  
**_X11b_train.npz_**: 另一部分訓練資料  
**_dataloader.py_**: 處理和轉換資料的程式碼(附上的)  

## Lab 3
### 概念: ResNet(套pretrain過的模型)
#### _Data太大不方便傳上來，請自行去spec網址下載_
**_DL_LAB3.ipynb_**: 程式碼  
**_DL_LAB3_demo.ipynb_**: demo用的程式碼  
**_Lab3-Diabetic-retinopathy-detection.pdf_**: spec  
**_Report_**: 報告  
**_dataloader.py_**: 讀取資料並整理成資料集的程式碼，附上的但原本有挖空  
**_test_img.csv_**: 測試資料的圖片編號  
**_test_label.csv_**: 測試資料的真正類別  
**_train_img.csv_**: 訓練資料的圖片編號  
**_train_label.csv_**: 訓練資料的真正類別  
**_train_mean.pt_**: 訓練資料圖片張量的平均值  
**_train_std.pt_**: 訓練資料圖片張量的標準差

## Lab 4
### 概念: cVAE(有條件的VAE)、LSTM、CNN
#### _Data太大不方便傳上來，請自行去spec網址下載_
#### 這次作業training要花超久(用課堂server的1000系列gpu跑一次就要花2天)，建議早點開始
**_models_**: 附上的，裡面包含2個檔案，總共4種模型  
* **_lstm.py_**: 兩種LSTM模型，一種是用在Encoder和Decoder中間叫做**Frame Predictor**，另一種是用在**Prior**  
* **_vgg_64.py_**: 兩種CNN類模型，一種是**Encoder**，另一種是**Decoder**  

**_CVAE參考論文.pdf_**: 附在spec的reference中，有關用fixed prior和learned prior(加分題，但時間不夠我沒做)做video prediction，強烈建議看一下，會比較知道要幹嘛  
**_Lab4_Conditional_VAE_for_Video_Prediction.pdf_**: spec  
**_Report.pdf_**: 報告  
**_dataset.py_**: 讀取並整理成資料集的程式碼(附上的)  
**_evaluation.py_**: 自己寫的測試模型程式碼，會印出幾次測試平均的psnr值，以及生成隨機一筆測試影片的預測gif以及圖片序列  
**_evaluation_utils.py_**: 模型測試時，會用到的一些工具函數，上面這個_evaluation.py_需要import這檔案  
**_kl_annealing.py_**: 模型測試時，把KL annealing的class單獨拉出來寫成一個檔案，然後_evaluation.py_也需要import這檔案  
**_train_fixed_prior.py_**: cVAE基於fixed prior來預測影片，本次作業主程式，附上但多處原本挖空  
**_utils.py_**: 工具函數，附上的但有些函數原本沒給要自己刻

## Lab 5
### 概念: GAN(架構、模型、損失函數全自己挑)
#### _Data太大不方便傳上來，請自行去spec網址下載_
#### 這次作業很難表現的很好:(...
**_Lab5_Let_s_Play_GANs.pdf_**: spec  
**_Report.pdf_**: 報告  
**_checkpoint.pth_**: _evaluator.py_ 中的評估模型的pretrained參數(附上的)  
**_dataset.py_**: 讀取和整理成資料集的程式碼  
**_evaluator.py_**: 測試階段使用。需要call這個class中"eval"這個方法，把模型生成的圖片和真正標籤丟進去，就會回傳準確率(附上的)  
**_main.py_**: 本次作業主程式  
**_model.py_**: GAN中的**Generator**和**Discriminator**的模型架構  
**_new_test.json_**: 測試資料的文字描述，期限前幾天才會公布(附上的)  
**_objects.json_**: 把顏色形狀描述的文字轉換成數字標籤(附上的)  
**_readme.txt_**: 有關_objects.json_、_train.json_、_test.json_ 的說明(附上的)  
**_test.json_**: 評估資料的文字描述(附上的)  
**_train.json_**: 訓練資料的文字描述(附上的)。訓練資料除了文字描述，也會有真正的圖片  
**_utils.py_**: 工具函數，用來讀取_test.json_或是_new_test.json_

## Lab 6
### 概念: 
