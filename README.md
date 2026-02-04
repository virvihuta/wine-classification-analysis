# Wine Quality Analysis & Classification

This project analyzes the **WineQT dataset** to explore physicochemical properties of wines and builds a machine learning pipeline to classify wine quality. The notebook moves from data exploration and visualization to binary classification (Good vs. Bad wine) using ML & deep learning techniques.

## Project Overview
The goal of this project is to understand the factors contributing to wine quality and predict whether a wine is considered "good" or "bad" based on its chemical features.

* **Input**: Physicochemical tests (e.g., fixed acidity, volatile acidity, alcohol, pH).
* **Output**: Binary quality classification (`Good` >= 6, `Bad` < 6).
* **Methodology**: Data preprocessing, Exploratory Data Analysis (EDA), and Neural Network modeling (PyTorch).

## Dataset
The project uses the `WineQT.csv` dataset located in the `data/` directory.
* **Features**: 11 input variables (e.g., density, sulphates, alcohol).
* **Target**: `quality` (score between 0 and 10), transformed into a binary `wineQuality_ord` target.

## Dependencies
This project requires Python and the following libraries:
* **Pandas & NumPy**: Data manipulation and numerical operations.
* **Matplotlib & Seaborn**: Data visualization (Pie charts, distribution plots).
* **PyTorch**: Building and training the neural network.

## Notebook Workflow

1.  **Data Loading & Inspection**:
    * Importing `WineQT.csv`.
    * Checking for missing values and data types.
    * Statistical summary.

2.  **Feature Engineering**:
    * Created a new categorical column `wineQuality` based on the score (threshold: 6).
    * Mapped categories to numerical values: `0` (Bad) and `1` (Good).

3.  **Exploratory Data Analysis (EDA)**:
    * Visualized the class balance using pie charts (approx. 54% Good vs. 46% Bad).
    * Analyzed feature distributions.

4.  **Model Architecture**:
    * Implementation of a Neural Network using **PyTorch**.
    * The model utilizes tensors on available acceleration devices (e.g., MPS/CUDA).

## Usage
1.  Clone the repository.
2.  Place `WineQT.csv` in a folder named `data/`.
3.  Run the Jupyter Notebook:
    `jupyter notebook main.ipynb`

## Results
* The dataset is relatively balanced between high and low-quality wines.
* The neural network weights indicate the model has been trained to discern non-linear relationships between chemical properties and wine quality.

***
