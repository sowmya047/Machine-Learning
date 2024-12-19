from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


class Ui_DetectFakeNewsApp(object):
    def browse_file(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File")
        self.lineEdit.setText(fileName)

    def browse_file1(self):
        fileName1, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File")
        self.lineEdit_2.setText(fileName1)

    def detect_fake_news(self):
        try:
            training_file = self.lineEdit.text()
            testing_file = self.lineEdit_2.text()

            if not training_file or not testing_file:
                self.showMessageBox("Information", "Please select both training and testing files.")
                return

            print("\nStep 1: Loading and Preprocessing Data...")
            train_data, test_data = self.load_and_preprocess_data(training_file, testing_file)

            print("\nStep 6: Training and Evaluating Models...")
            results = {}
            results['SVM'] = self.train_and_evaluate_svm(train_data, test_data)
            results['Logistic Regression'] = self.train_and_evaluate_lr(train_data, test_data)
            results['LSVM'] = self.train_and_evaluate_lsvm(train_data, test_data)

            print("\nStep 7: Displaying Results...")
            self.display_results(train_data, test_data, results)

        except Exception as e:
            print(f"Error: {str(e)}")

    def load_and_preprocess_data(self, train_file, test_file):
        print("Step 1: Loading datasets...")
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        print(f"Training data loaded: {train_data.shape[0]} rows, {train_data.shape[1]} columns")
        print(f"Testing data loaded: {test_data.shape[0]} rows, {test_data.shape[1]} columns")

        print("\nStep 2: Filling in missing values...")
        train_data['Statement'] = train_data['Statement'].fillna('')
        train_data['Label'] = train_data['Label'].fillna('unknown')
        test_data['Statement'] = test_data['Statement'].fillna('')
        test_data['Label'] = test_data['Label'].fillna('unknown')
        print(f"Missing values handled. Training data sample:\n{train_data.head()}")

        print("\nStep 3: Detecting outliers...")
        train_data['Length'] = train_data['Statement'].apply(len)
        test_data['Length'] = test_data['Statement'].apply(len)
        outlier_threshold = train_data['Length'].mean() + 3 * train_data['Length'].std()
        outliers = train_data[train_data['Length'] > outlier_threshold]
        print(f"Detected {len(outliers)} outliers in the training data. Outliers:\n{outliers[['Statement', 'Length']].head()}")
        train_data.drop(outliers.index, inplace=True)
        print(f"Outliers removed. Remaining training data: {train_data.shape[0]} rows.")

        print("\nStep 4: Normalizing data using MinMaxScaler...")
        scaler = MinMaxScaler()
        train_data['Normalized_Length'] = scaler.fit_transform(train_data[['Length']])
        test_data['Normalized_Length'] = scaler.transform(test_data[['Length']])
        print(f"Normalization applied. Training data sample:\n{train_data[['Length', 'Normalized_Length']].head()}")
        
        # Step 5: Applying PCA for dimensionality reduction
        print("Step 5: Applying PCA for dimensionality reduction...")
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Statement'])

        # Dynamically determine n_components
        n_components = min(300, tfidf_matrix.shape[0], tfidf_matrix.shape[1])
        print(f"Applying PCA with n_components={n_components}...")

        pca = PCA(n_components=n_components)  # Adjust n_components dynamically
        train_data_pca = pca.fit_transform(tfidf_matrix.toarray())
        test_data_pca = pca.transform(tfidf_vectorizer.transform(test_data['Statement']).toarray())
        print(f"PCA completed. Data reduced to {n_components} components.")

        train_data['PCA_Features'] = list(train_data_pca)
        test_data['PCA_Features'] = list(test_data_pca)

        return train_data, test_data

    def train_and_evaluate_svm(self, train_data, test_data):
        print("Training and Evaluating SVM...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(train_data['PCA_Features'].tolist(), train_data['Label'])
        pipeline = Pipeline([
            ('svm', svm.SVC(kernel='linear', class_weight='balanced'))
        ])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(test_data['PCA_Features'].tolist())
        accuracy = accuracy_score(test_data['Label'], predicted)
        print("SVM Classification Report:")
        print(classification_report(test_data['Label'], predicted))
        return accuracy

    def train_and_evaluate_lr(self, train_data, test_data):
        print("Training and Evaluating Logistic Regression...")
        pipeline = Pipeline([
            ('lr', LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced'))
        ])
        pipeline.fit(train_data['PCA_Features'].tolist(), train_data['Label'])
        predicted = pipeline.predict(test_data['PCA_Features'].tolist())
        accuracy = accuracy_score(test_data['Label'], predicted)
        print("Logistic Regression Classification Report:")
        print(classification_report(test_data['Label'], predicted))
        return accuracy

    def train_and_evaluate_lsvm(self, train_data, test_data):
        print("Training and Evaluating Linear SVM...")
        pipeline = Pipeline([
            ('lsvm', svm.LinearSVC(class_weight='balanced', max_iter=2000))
        ])
        pipeline.fit(train_data['PCA_Features'].tolist(), train_data['Label'])
        predicted = pipeline.predict(test_data['PCA_Features'].tolist())
        accuracy = accuracy_score(test_data['Label'], predicted)
        print("Linear SVM Classification Report:")
        print(classification_report(test_data['Label'], predicted))
        return accuracy

    def display_results(self, train_data, test_data, results):
        print("\n--- Classification Results ---")
        for model, accuracy in results.items():
            print(f"{model}: Accuracy = {accuracy:.2f}")

        results_text = "\n".join([f"{k}: {v:.2f}" for k, v in results.items()])
        self.showMessageBox("Results", f"Results:\n{results_text}")

    def showMessageBox(self, title, message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
        msgBox.setWindowTitle(title)
        msgBox.setText(message)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.exec_()

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(800, 600)
        Dialog.setWindowTitle("Fake News Detection")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(150, 50, 500, 50))
        self.label.setText("Fake News Detection Using Machine Learning")
        self.label.setStyleSheet("font: 18pt 'Arial'; color: #333; text-align: center;")

        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(150, 150, 500, 30))

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(670, 150, 100, 30))
        self.pushButton.setText("Browse")
        self.pushButton.clicked.connect(self.browse_file)

        self.lineEdit_2 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_2.setGeometry(QtCore.QRect(150, 200, 500, 30))

        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(670, 200, 100, 30))
        self.pushButton_2.setText("Browse")
        self.pushButton_2.clicked.connect(self.browse_file1)

        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(300, 300, 200, 50))
        self.pushButton_3.setText("Detect Fake News")
        self.pushButton_3.clicked.connect(self.detect_fake_news)

        QtCore.QMetaObject.connectSlotsByName(Dialog)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_DetectFakeNewsApp()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
