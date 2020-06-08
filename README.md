# Pneumonia Diagnosis
A Convolutional Neural Network that is able to detect whether a patient has pneumonia, both bacterial and viral, or not, based on an X-ray image of their chest. Implements transfer learning, using the first 16 layers of a pre-trained VGG19 Network, to identify the image classes.


To run this project, follow the steps below:
1. Install the Kaggle API, which will assist in the downloading of data:
```
pip install kaggle
```
2. Retrieve the data through the following command:
```
python retrieve.py
```
3. Train the Convolutional Neural Network through the following command:
```
python train.py
```
4. Make predictions on new, unseen chest X-ray using the Convolutional Neural Network The default image is 'test.jpg', which does possess Pneumonia:
```
python predict.py -i <path-to-image>
```


By completing the following steps, the following files are created:
* model.h5 - This stores a '.h5' version of the Convolutional Neural Network model trained.
* plot.jpg - This displays statistics regarding the training process of the model.


The final accuracy obtained by the model, after testing on 624 unseen instances, is approximately 92%. 
