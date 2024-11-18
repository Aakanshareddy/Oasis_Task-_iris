# Oasis_Task-_iris

# Iris Flower Classification Project

This project classifies Iris flowers into three species: **Iris-setosa**, **Iris-versicolor**, and **Iris-virginica** using the **K-Nearest Neighbors (KNN)** algorithm.

## **Project Highlights**
1. **Dataset**: 
   - Used the Iris dataset containing 150 samples with 4 features:
     - **SepalLengthCm**
     - **SepalWidthCm**
     - **PetalLengthCm**
     - **PetalWidthCm**
   - Target variable: **Species**

2. **Libraries Used**:
   - `pandas`: Data manipulation.
   - `matplotlib.pyplot` and `seaborn`: Data visualization.
   - `scikit-learn`: Machine learning model implementation.

3. **Data Insights**:
   - Descriptive statistics:
     - Sepal length: Min = 4.3, Max = 7.9.
     - Petal length: Min = 1.0, Max = 6.9.
   - Visualized relationships between features using a **pair plot**, distinguishing species by color.

4. **Data Preprocessing**:
   - Removed unnecessary columns like `Id`.
   - Features (X): `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`.
   - Target (Y): `Species`.
   - Split dataset into **training (70%)** and **testing (30%)** subsets.

5. **Model Training**:
   - Algorithm: **K-Nearest Neighbors (KNN)**.
   - Hyperparameters:
     - `n_neighbors=3`.
   - Training accuracy: **94.3%**.
   - Testing accuracy: **100%**.

6. **Model Evaluation**:
   - Achieved **perfect accuracy** (1.0) on the test set.
   - Classification Report:
     - **Precision, Recall, F1-Score**: 1.00 for all species.
   - Confirms the model's robustness and ability to generalize.

7. **Predictions**:
   - New data predictions:
     - Example 1:
       ```python
       SepalLengthCm: 5.1, SepalWidthCm: 3.5, PetalLengthCm: 1.4, PetalWidthCm: 0.2
       Prediction: Iris-setosa
       ```
     - Example 2:
       ```python
       SepalLengthCm: 5.1, SepalWidthCm: 9.5, PetalLengthCm: 9.4, PetalWidthCm: 0.2
       Prediction: Iris-virginica
       ```

8. **Key Metrics**:
   - **Train/Test Data Split**:
     - Training size: (105, 4)
     - Testing size: (45, 4)
   - **Performance Metrics**:
     - Accuracy: **1.0**.
     - Detailed metrics (precision, recall, F1-score) all equal to **1.0** for each class.

## **Code Summary**
1. **Data Loading and Visualization**:
   ```python
   dataset = pd.read_csv('/content/Iris.csv')
   sns.pairplot(dataset, hue="Species")
   plt.show()
   ```

2. **Feature Selection and Splitting**:
   ```python
   X = dataset.drop(['Id', 'Species'], axis=1)
   Y = dataset["Species"]
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
   ```

3. **Model Training**:
   ```python
   knn = KNeighborsClassifier(n_neighbors=3)
   knn.fit(X_train, Y_train)
   ```

4. **Evaluation**:
   ```python
   print(f'Training Accuracy: {knn.score(X_train, Y_train)}')
   print(f'Testing Accuracy: {knn.score(X_test, Y_test)}')
   print(classification_report(Y_test, knn.predict(X_test)))
   ```

5. **Prediction**:
   ```python
   new_data = pd.DataFrame({"SepalLengthCm": [5.1], "SepalWidthCm": [3.5], "PetalLengthCm": [1.4], "PetalWidthCm": [0.2]})
   prediction = knn.predict(new_data)
   print(prediction[0])  # Output: Iris-setosa
   ```

## **Conclusion**
This project demonstrates the application of KNN for classifying Iris flower species with high accuracy, making it a great foundational project for understanding supervised machine learning techniques.
