# ACSE-8-Mini-Project
Keer Mei | Yujie Zhou | Adanna Akwataghibe | Tayfun Karaderi
## Team Entropy - Kuzushiji-MNIST Classification

The Kuzushiji-MNIST (or KMNIST)  is similar to the MNIST dataset with 28x28 grayscale, 70,000 images, labelled into 10 classes. This data set consists of the *Kuzushiji* cursive Japanese characters. These characters are not used in writing or taught in schools since the modernisation of the Japanese language. However, there are classical literature written before this modernisation that cannot be read by the average Japanese. Therefore, deep learning approaches are used to come up with ways to classify these characters with a high enough accuracy, so that these classical books can be read [[1]([https://arxiv.org/pdf/1812.01718.pdf](https://arxiv.org/pdf/1812.01718.pdf))]. 

This project aims to produce a neural network that best classifies the images in the KMNIST dataset.  

## Repository Format:
**KMNIST_ENTROPY/ :** Folder that contains all the notebooks which have the models we used and the functions used to build and optimise them. 

In the folder, there are the following:
- **models/**
	- contains the pre-trained models (AlexNet or LeNet), saved as *pt* files. 
- **results/**
	- contains the predictions of the testset, saved as *csv* files. 
- **AlexNet.ipynb**
	- Contains the AlexNet model and steps to train and save your own model or reproduce our optimal predictions. 
- **Evaluate_Model.ipynb**
	- Contains functions that loads pre-trained models and returns their accuracy scores. 
- **LeNet.ipynb**
	- Contains LeNet model and steps to train and save your own model or reproduce our optimal predictions. 
	- **Kaggle Submission**: 
		- *Model:* LeNet5_pluslayer_kmnist_classifier_random_aug_20_choice_5kernel_ep_10_complete_training_set_extended.pt
		- *Result:* LeNet5_kmnist_classifier_random_aug_20_choice_5_kernel_ep_10_complete_extended.csv
- **Model_Averaging.ipynb**
	- Contains model averaging functions and steps combine your own models or to reproduce our optimal combinations. 
	- All the current files in the **results/** folder were used in the model prediction combination function to obtain the optimal predictions. 
	- Additionally, the models in the **models/** folder were used to create these predictions. 
	- **Kaggle Submission**
		- *Result:* Lenet5_extended_Alex5_aug_20c_ep10_ep15_.csv
- **PCA_Classifier.ipynb**
	- Contains PCA Classifier that uses PCA to reduce the dimensionality of the dataset and one of the following classifiers 
		- K-Nearest Neighbours 
		- Random Forrest (*not in lectures*)
		- Naive Bayesian (*not in lectures*)
- **Utils.ipynb**  
	- Contains the functions needed for training, validation, evaluation, k-fold cross validation, data augmentation etc. This notebook is essential as all of the other notebooks rely on it.
- **\__init\__.py**
	- Used for making the python notebooks a module

## Prerequisites
**Note:** 
1. Download the folder **KMNIST_ENTOPY/** and add it to your Google drive (in *My Drive*), so the path in should *My Drive/KMNIST_ENTROPY*. 
2. GPU in the google collab needs to be set in the hardware accelerator. Our notebooks currently only function for this runtime type.
3. If you change any function in the Utils.ipynb, then you will need to *restart runtime* in the model notebook that you are using to classify. 

## Requirements 
- Python 3

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Report 

The full research report is in the file **kmnist_team_entropy_final_report.pdf**
