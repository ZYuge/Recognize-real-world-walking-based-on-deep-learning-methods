This algorithm was developed by Y. Zhang, Sina David, Sjoerd M. Bruijn, Michiel Punt, Jorunn L. Helbostad and Mirjam Pijnappels from Vrije Universiteit Amsterdam, the Utrecht University of Applied Sciences, and Norwegian University of Science and Technology.  


# Recognize real-life gait episodes using lower back IMU for older adults

We developed a convolutional neural network (CNN) based on inertial sensor data of the lower back (L5) to classify real-life activities in two categories, gait and non-gait, as presented in paper **[1] [link](https://link.springer.com/article/10.1007/s11517-025-03466-z)**.  

This model is suitable for older people who walk slowly. The data for model training came from healthy and gait-impaired older adults with mean age 76.4(5.6) years old [2]. The data for externally validation came from stroke survivors who could walk independently at a mean age of 72.4 (12.7) years [3]. 


## 1. How to use the algorithm

Before running the code, create a virtual environment with Python 3.11 or 3.12 and install the checked dependencies from `requirements.txt`:  

```bash
python -m venv .venv
source .venv/bin/activate  # macOS or Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`os`, `re`, `pickle`, and `math` are part of Python's standard library and do not need to be installed separately.  

We provide the code for the whole process, including **data preprocessing** (data reading, balancing, augmentation and segmentation), **model training (Aim 1)** (model evaluation, fit,  and overfitting prevention), and **external validation process (Aim 2)** (data reading, model prediction, model performance evaluation).  

We also provide the best-performing model that we have obtained, into which you can put your sensor data to get the binary results **(Aim 3)**.   

In **"main_code.py"**, you can call the functions in **"GaitRecognitionFunctions.py"** for following different aims.  

```
import GaitRecognitionFunctions as GR
```


**1) train a CNN model**  
```
DataX, DataY_binary, groups, filenames, all_subjects = GR.load_matfiles(InputDataDir, aim_label, input_axis, del_labels)

GR.train(repeats, DataX, DataY_binary, groups, Nfolds, augmentation_methods, input_axis, Nfolds_val, fs_trainingdata,
          window_size, overlap_rate, ModelsInfoDir, ModelsSaveDir, ModelResultDir)
```

The `GR.train()` is responsible for data preprocessing, repeated holdouts-validation, data augmentation, model training, and model saving. It controls the model training process and saves the results through a series of input parameters.  

`repeats`: number of training repetitions. In each traing, the dataset split and kernel weight in the model are random.   
`DataX`: the sensor data for model training (6 channels: acceleration and gyroscope data).    
`DataY_binary`: the binary activity label corresponding to each sampling point in DataX, 'walking=1, non-walking=0'.    
`groups`: the subject numbers corresponding to each sampling point.  
Other function input can be seen in below 4)


**2) externally validate the existing model**
```
DataX, DataY, groups = GR.load_txtdata(ExValDataTxtDir, input_axis)

GR.validation(fs_validationdata, window_size, overlap_rate, input_axis, model_path, DataX, DataY, groups,
               augmentation_methods, plotsingal, ExValScoresDir)
```
This `GR.validation()` is responsible for loading an existing pre-trained model, visualizes its structure, loading external validation data, and performing model using the specified settings, such as window size, overlap rate, and augmentation methods. The results and plots are stored in the directory `ExValScoresDir`.

`DataX`, `DataY`,`groups` are based on external validation data. Here, since external validationi data we used is only consist of walking and standing, we haven't use `GR.dichotomy_labels()` to get DataY_binary. If your external validation data is not from binary activities, you need to add it in `GR.load_txtdata()`.  


**3) predict the unknown activities**
```
GR.predict_data_unsupervised(X, model_path, window_size, overlap_rate)
```

`X`: the unsupervised input data for the prediction, which can be acceleration only, or both aceeleration and gyroscope.  
`model_path`: the path of responding model for data of 3 channels or 6 channels.  Models in './best_cnn_models' can be used.    
`window_size`: the same with the one in model training.  
`overlap_rate`: the same with the one in model training.  

**4) the general settings in the begining**  
`wk_label`: the defined label of walking.  
`aim_label`: the responding label of our targeted activity.   
`del_label`: the activity labels can be deleted to reduce computation, such as 'undefined' and 'none'.  
`window_size`: the winidow size of data in model training.  
`overlap_rate`: the overlap rate in each window.  
`Nfolds`: the number of partitions (folds) you want to divide the dataset into training and testing datasets by subjects for repeated holdout-validation.  
`augmentation_methods`: choose the methods you want to use for augmentation, it can be none, one method or multiple methods. Here, we provide methods of 'noise','scaling', 'interpolation', 'rotation'.   
`input_axis`: the input channels, can be 3 for acceleration data only, and can be 6 for aceeleration & gyroscope data.   
`Nfolds_val`: the number of partitions (folds) you want to divide the training dataset into training and validating datasets.  
`fs_trainingdata`: the sampling frequency of model training data.  
`fs_validationdata`: the sampling frequency of externally validation data.   
`InputDataDir`: the folder storing the input data for model training.   
`ModelsInfoDir`: store subject numbers of splitted datasets and kernel weights in each training repeat.    
`ModelsSaveDir`: store model with '.h5' in each training repeat.  
`ModelsResultDir`: store the model performance in each training repeat.  
`ExValDataTxtDir`: store the input data for external validation. Here, the data was stored in '.txt'.   
`ExValScoreDir`: store the model performance on external validation, picture of true and predicted labels on signals.   


## 2. Format of data
The example data can be downloaded by using the link [UPLOAD EXAMPLE FILE](https://drive.google.com/drive/folders/1IzSRh8Th3zzxsP3cEAWY3xVwbSjO4WhD?usp=sharing).
### 2.1 Input Data in matfiles 

We used **ADAPT dataset [2] for model training**, including semi-structured supervised and real-life unsupervised situation of 20 older adults. All activities are labeled synchronously. 

**The example data is stored in the dir './github/Example data/for model training_Aim1'. The path of it can be set as `InputDataDir` in the code.**

In the folder `InputDataDir`, each subject's data is stored as a '.mat' file with subject number. The sensor data is as a matrix 'signal' in the mat file, where responding columns are 3-axis acceleration, 3-axis gyroscope, 3-axis magnetometer data, activity labels and time. XYZ axes respond to vertical (up, +), medial-lateral (right +), and anteroposterior (anter +), respectively. 

After loading each subject's data and appending it cumulatively, we get DataX, DataY, DataY_binary and groups.The function we use is `GR.load_matfiles()`.
- DataX has 6 axes, i.e., 3-axis acceleration, 3-axis gyroscope.  
- DataY includes all activity labels.  
- DataY_binary includes labels of walking(=1) and non-walking(=0). 
- Groups list the subject numbers corresponding to each sampling point.

The activity labels of **ADAPT dataset** represent:
0=none, 1=walking, 2=walking with transition, 3=shuffling,  
4=stairs(ascending), 5=stairs(descending), 6=standing, 
7=sitting, 8=lying, 9=transition, 10=leaning, 11=undefined,  
12=jumping, 13=dynamic, 14=static, 15=shake, 16=picking, 17=kneeling.

### 2.2 Input Data in .txt files 

The original data is collected in '.txt' files of each individual for walking and non-walking, with 6 columns in each file for acceleration and gyroscope data. And the activity is also listed in the filenames. After orgnizing, all subjects's signals are spliced vertically, so the same to y labels and groups.

As shown in the folder `ExValDataTxtDir`, we have 6 files for **external validation data**. The function we use is `GR.load_txtdata()`.
- signals_wk.txt
- signals_nonwk.txt
- y_wk.txt
- y_nonwk.txt
- groups_wk.txt
- group_sta.txt


## 3. Data augmentation
The function `data_augmentation_general.py` contains the following options for methods. The hyperparameters of them are show as below and you can modify them by your own: 
| augmentation methods       | recommended scale |
| -------------------------- | ----------------- |
| Jitter                     | [0.01, 0.02]      |
| scaling                    | [0.3,0.4]         |
| Resampling (interpolation) | [0.9,1.1]         |
| Rotation                   | [90°]             |

The function is called in `main_code.py`for model training and external validation as below:
```
import data_augmentation_general.py as DA

GR.train(repeats, DataX, DataY_binary, groups, Nfolds, augmentation_methods, input_axis, Nfolds_val, fs_trainingdata,
          window_size, overlap_rate, ModelsInfoDir, ModelsSaveDir, ModelResultDir)

GR.validation(fs_validationdata, window_size, overlap_rate, input_axis, model_path, DataX, DataY, groups,
               augmentation_methods, plotsingal, ExValScoresDir)
```


![Model Structure](images/Model%20Structure.png)
**Figure1. CNN Model structure (The input IMU data can be 3-axis or 6-axis, W is the window size, 200 is 2 seconds with sampling frequency 100 Hz)**  
(Hyperparameters: Epochs = 30, Batch_size = 32, Filters = 64, Kernel_size = 3)  

![Model performance_external dataset](/images/Model%20performance_external%20dataset.png)
**Figure2. CNN Model performance on external dataset (stroke patients)(Note:DA, the abbreviation for data augmentation, here we use rotation 90° on xyz-axis, separately）**


## 4. References

[1] [Our ADAPT paper after publishing](https://link.springer.com/article/10.1007/s11517-025-03466-z)  
[2] Bourke AK, Ihlen EAF, Bergquist R, Wik PB, Vereijken B, Helbostad JL. A physical activity reference data-set recorded from older adults using body-worn inertial sensors and video technology—The ADAPT study data-set. Sensors. 2017;17(3):559.  
[3] Felius RAW, Geerars M, Bruijn SM, et al. Reliability of IMU-Based Gait Assessment in Clinical Stroke Rehabilitation. Sensors (Basel) 2022;22(3).  

 


## Help

For questions about the algorithm and the implementations please contact: [y6.zhang@vu.nl](mailto:y6.zhang@vu.nl)
