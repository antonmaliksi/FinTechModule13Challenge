# Deep Learning Predictive Model

---

To the business team at Alphabet Soup, I present a binary classification model using a deep neural network along with two alternative models, each with their own benefits and drawbacks. These models will predict whether applicants will be succesfull if we were to provide support.

Using the CSV file provided, I have performed the following:

1. Pre-processed the data for a neural network model.
2. Used the model-fit-predict pattern to compile and evaluate a binary classification model.
3. Optimized the model.

![header](url)

---

## Technologies

This notebook utilizes **Python (v 3.9.7)** and the following libraries and dependencies:

1. pandas
2. Path from pathlib
3. tensorflow
4. Dense from tensorflow keras
5. Sequential from tensorflow keras
6. train_test_split from sklearn
7. StandardScalar and OneHotEncoder from sklearn

---

## Installation Guide
Pandas and Pathlib should be part of the base applications that were installed with the Python version above; if not, you will have to install them through the pip package manager of Python.

To install TensorFlow, run the following:

    ```
    pip install --upgrade tensorflow
    ```
    
Once the TensorFlow installation is complete, verify the installation by running the following:

    ```
    python -c "import tensorflow as tf;print(tf.__version__)"
    ```
  
The output of the previous command should show version 2.0.0 or higher.

Keras is intrinsically part of TensorFlow, so run the following to verify that the package is installed:

    ```
    python -c "import tensorflow as tf;print(tf.keras.__version__)"
    ```

If any errors occur, please contact IT for further assistance.

---

## User Guide

To use the notebook:

### Load the Data
1. Open "venture_funding_with_deep_learning.ipynb"
2. Look for the following code:
    ```python
    applicant_data_df = pd.read_csv(Path("Resources/applicants_data.csv"))

    applicant_data_df.head()
    ```
Please ensure that you have the correct CSV file in ```(Path("Resources/file.csv"))``` located within the Resources folder.

### Encode the Data
Before conducting our analysis and constructing the deep learning model, we have to encode our data. <br> We will do this by using sklearn's OneHotEncoder function:
1. Create a list of categorical variables.
2. Create a OneHotEncoder instance:
![OHE](url)
We use ```(sparse=False)``` to fetch a NumPy array which is used to supplement the OneHotEncoder instance.
3. Encode the categorical variables using OneHotEncoder.
4. Create a DataFrame with the encoded variables.
5. Separate our data into X and y variables for machine learning:
![Xy](url)
6. Use ```train_test_split()``` to split our data into two training and testing sets.
7. Use ```StandardScaler()``` to scale our data.
8. Fit and transform our scaled training and testing data.

### Compile and Evaluate a Binary Classification Model Using a Neural Network
To begin creating our deep learning neural network, we have to determine:
1. The number of inputs.
2. The number of hidden layers.
3. The number of neurons in each layer.
4. The activation method of each hidden layer.
5. The number of outputs.
6. The activation method, optimizer, and additional metrics for the output(s).
For our model, we will use the following parameters:
![parameters](url)
Some of these numbers, however, are not arbitrary. Using our scaled data, we see that we have 116 features which will be used as the number of inputs.
7. Use sklearn's ```Sequential()``` function to create our model's instance.
8. Use sklearn's ```Dense()``` to build the framework of our model:
![model](url)
9. Compile our model:
    ```python
    nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["Accuracy"])
    ```
10. Fit the scaled training and testing data to our model and set the amount of epochs to be used:
    ```python
    model = nn.fit(X_train_scaled, y_train, epochs=50)
    ```

### Evaluate and Save the Model
![evsave](url)

---

## Alternative Models
With our original model, we were able to determine our model's accuracy to 73% and a loss factor of 55%. <br> While this is a good start, we want to further optimize our model to find the highest amount of accuracy <br> and lowest amount of loss. To do this, we can do the following:

1. Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
2. Add more neurons (nodes) to a hidden layer.
3. Add more hidden layers.
4. Use different activation functions for the hidden layers.
5. Add to or reduce the number of epochs in the training regimen.

### Alternative Model 1
For our first optimization, we decreased the amount of neurons in layer 1 to 50, decreased the amount of neurons in layer 2 to 20, and added a third layer with 10 neurons and the same activation.

As a result, we increased our accuracy from .7314 to .7322 and reduced our losses from .5591 to .5523. A great success! But we can do better.

### Alternative Model 1
Utilizing the findings from our first optimization, we decided to keep all of the parameters of the <br> model the same and add 10 extra epochs; This thinking was to see if adding more time <br> would allow the model to reach a higher accuracy and lower loss. <br> What we found surprised us, as our accuracy decreased and loss increased - not at all optimal or benefitial to our mission. <br> Although this model was not optimal, we want to showcase how adding extra epochs might not always be the smartest choice for the given situation. <br> Other options included creating a fourth hidden layer and decreased the amount of neurons across <br> the model, as well as using different activation methods.

### Comparing the Three Models.
![models](url)

---

## Versioning History
All GitHub commits/pulls were conducted and verified by Risk Management Associate Anton Maliksi.

---

## Contributors
Anton Maliksi was the sole contributor for this notebook, with collaboration from James Handral.

---

## Licenses
No licenses were used for this notebook.