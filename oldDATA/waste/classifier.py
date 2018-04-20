from sklearn import datasets
from sklearn.model_selection import train_test_split
import csv

# X - training data, y - classes
X = []
y = []

# Using iris dataset
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target

# loading csv file
#dataset = 'dataset.csv'
# with open('dataset.csv', 'rb') as csvfile:
#    reader = csv.reader(csvfile)
#    for row in reader:
#        X.append([row[0],row[1]])
#        y.append(row[2])
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_rbf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
svm_predictions = svm_model_rbf.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_rbf.score(X_test, y_test)
print(accuracy)
