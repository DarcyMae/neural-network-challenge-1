# Student Loan Risk Assessment with Deep Learning

## Project Overview
This project uses deep learning techniques to assess the risk of student loan defaults. It utilizes a neural network model built with TensorFlow and Keras to predict the likelihood of a student successfully repaying their loan based on various factors.

## Features
- Data preprocessing and feature scaling
- Neural network model creation using TensorFlow and Keras
- Model training and evaluation
- Prediction on test data
- Classification report generation

## Requirements
- Python 3.10+
- TensorFlow
- Pandas
- NumPy
- Scikit-learn

## Installation
To set up the project environment, run:
```
pip install tensorflow pandas numpy scikit-learn
```

## Usage
1. Load and preprocess the data:
   ```python
   loans_df = pd.read_csv("student-loans.csv")
   X = loans_df.drop("credit_ranking", axis=1)
   y = loans_df["credit_ranking"]
   ```

2. Split the data and scale features:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

3. Create and compile the model:
   ```python
   model = Sequential([
       Dense(16, activation="relu", input_dim=X_train.shape[1]),
       Dense(8, activation="relu"),
       Dense(1, activation="sigmoid")
   ])
   model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
   ```

4. Train the model:
   ```python
   history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
   ```

5. Evaluate the model:
   ```python
   loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)
   print(f"Loss: {loss}, Accuracy: {accuracy}")
   ```

6. Make predictions:
   ```python
   predictions = model.predict(X_test_scaled)
   predictions_df = pd.DataFrame({"predictions": predictions.round()})
   ```

7. Generate classification report:
   ```python
   print(classification_report(y_test, predictions_df["predictions"]))
   ```

## Model Performance
The model achieves approximately 75% accuracy in predicting student loan repayment success.

## Future Improvements
- Feature engineering to potentially improve model performance
- Hyperparameter tuning
- Exploration of other machine learning algorithms for comparison

## License
This project is open-source and available under the MIT License.
