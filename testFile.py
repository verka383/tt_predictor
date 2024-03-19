from sklearn import tree

# Define the dataset
X = [[0, 0], [0, 1], [1, 0], [1,1]]  # input features
y = [0, 1, 1, 0]  # corresponding output labels

# Create the classifier and fit it to the data
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Use the classifier to predict new instances
print(clf.predict([[0, 0], [0, 1], [1, 0], [1, 1]]))  # prints [0 1 1 0]