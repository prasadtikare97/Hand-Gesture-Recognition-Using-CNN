Gesture recognition is an active research field in Human-Computer Interaction technology. It has many applications in virtual environment control and sign language translation, robot control, or music creation. 
In this machine learning project on Hand Gesture Recognition, we are going to make a real-time Hand Gesture Recognizer using the MediaPipe framework and Tensorflow in OpenCV and Python.
OpenCV is a real-time Computer vision and image-processing framework built on C/C++. But we’ll use it on python via the OpenCV-python package.

What is MediaPipe? 
MediaPipe is a customizable machine learning solutions framework developed by Google. 
It is an open-source and cross-platform framework, and it is very lightweight. 
MediaPipe comes with some pre-trained ML solutions such as face detection, pose estimation, hand recognition, object detection, etc.

What is Tensorflow? 
TensorFlow is an open-source library for machine learning and deep learning developed by the Google brains team. It can be used across a range of tasks but has a particular focus on deep neural networks.
Neural Networks are also known as artificial neural networks. It is a subset of machine learning and the heart of deep learning algorithms. The concept of Neural networks is inspired by the human brain. 
It mimics the way that biological neurons send signals to one another. Neural networks are composed of node layers, containing an input layer, one or more hidden layers, and an output layer.

We’ll first use MediaPipe to recognize the hand and the hand key points. MediaPipe returns a total of 21 key points for each detected hand. We are going to recognize hand gestures from a video sequence. 

File Description
Collect_all_images.py - Run this file to generate for gathering and organizing image data from various sources into a dataset. 
Create_train_dataset.py - Run this file to preprocess and partition the collected images into training and possibly validation sets for model training. 
ModelTraineripy - This is the model trainer file. Run this file if you want to retrain the model using your custom dataset. 
Gestureclassifier.py - Run this file to implement the gesture classification model, including functions for loading the model, performing predictions, and handling input data.
