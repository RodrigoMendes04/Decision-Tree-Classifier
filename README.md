# 🌳 Decision Tree Classifier - Python 🌳

## 📚 Description

Welcome to the **Decision Tree Classifier** implementation in Python! 🎉 This repository contains a simple, yet powerful, Decision Tree model for **supervised classification** tasks. It’s designed to help you understand how decision trees work by implementing them from scratch, without relying on specialized libraries like Scikit-learn.

Decision trees are popular machine learning models that split data based on features to make predictions. This project includes:
- 📊 Building a **decision tree** using **information gain** and **entropy**.
- 🤖 **Classifying new data** based on the trained decision tree.
- 📈 **Training and testing** the model on different datasets.
- 🎨 **Visualizing** the decision tree in a textual format.

### 🚀 Features:
- **Recursive Tree Construction**: The decision tree is built recursively, splitting the dataset based on the most relevant features.
- **Classification**: Once trained, the model can classify new instances based on learned patterns.
- **Tree Visualization**: The decision tree can be printed in a readable textual format for easier understanding.
- **Accuracy Calculation**: Computes the accuracy of the model based on real vs. predicted values.
- **Supports Different Data Types**: Works with both numerical and categorical data.

### 🏃‍♂️ Usage
#### 🔥 Train the Decision Tree
To train the model using a dataset, simply run:
```bash
python main.py -tr 'dataset_name'
```
Replace 'dataset_name' with one of the available datasets:
- iris
- restaurant
- weather
- connect4

#### 🧪 Test the Model
After training the model, you can test it using another dataset:
```bash
python main.py -tr 'training_dataset' -t 'test_dataset'
```
Where:
- 'training_dataset' is the dataset used for training (e.g., iris).
- 'test_dataset' is the dataset used for testing (e.g., weather).

### 💡 Help Menu
Need help? Run the following command to display available options:
```bash
python main.py -h
```
This will show the following options:
- -tr, --train: The dataset used for training the model.
- -t, --test: The dataset used for testing the model.
- -h, --help: Displays the help menu.
