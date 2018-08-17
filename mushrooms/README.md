# Mushrooms classification

The objective of this project is to classify mushrooms into 'edible' and 'poisonous'. 
This project was originally a competition hosted by [Kaggle](https://www.kaggle.com/uciml/mushroom-classification).

The dataset for this project consist of labeled data instances of mushrooms features.
The model used to solve this task is a fed forward neural network composed by a stack 
of fully connected layers with non-linear activations, and a sigmoid classification layer.

There are 2 ways of using this project.

- ## Jupyter Notebooks

    There are 3 jupyter notebooks on this project, that try to show and explain every 
    step it takes to solve the problem. These are:
     - `mushrooms_data_preparation.ipynb`: It's used to explore the dataset. It also 
        shows how the data is processed before it can be feed into a neural network model. 
        Run this first to be able to use the other notebooks.
     - `mushrooms_tensorflow.ipynb`: Create a Tensorflow model step by step, and then 
        train, evaluate and test it on the project's dataset (previously processed). 
     - `mushrooms_keras.ipynb`: Create a Keras model step by step, and then 
        train, evaluate and test it on the project's dataset (previously processed). 

- ## Python scripts

    - ### mushrooms_tensorflow.py
        Process the data and create a Tensorflow model. Train the model and plot training 
        information. The basic usage of this script is:
        
        ```bash
        python mushrooms_tensorflow.py
        ```
        
        The model architecture, and the training hyperparameters can be changed using some 
        script arguments. For a full list of available arguments use:
        
        ```bash
        python mushrooms_tensorflow.py -h
        ```
        
        Let's say we want to train a model on 30 epochs, using a mini batch size of 64 and 
        a learning rate of 0.1. We will need to run the script like this:
        
        ```bash
        python mushrooms_tensorflow.py --epochs 30 --batch_size 64 --learining_rate 0.1
        ```
    - ### mushrooms_keras.py
        Process the data and create a Keras model. Train the model and plot training 
        information. The basic usage of this script is:
        
        ```bash
        python mushrooms_keras.py
        ```
        
        The model architecture, and the training hyperparameters can be changed using some 
        script arguments. For a full list of available arguments use:
        
        ```bash
        python mushrooms_keras.py -h
        ```
        
        Let's say we want to train a model on 30 epochs, using a mini batch size of 64 and 
        a learning rate of 0.1. We will need to run the script like this:
        
        ```bash
        python mushrooms_keras.py --epochs 30 --batch_size 64 --learining_rate 0.1
        ```
