#from sklearn import tree

## Define the dataset
#X = [[0, 0], [0, 1], [1, 0], [1,1]]  # input features
#y = [0, 1, 1, 0]  # corresponding output labels

## Create the classifier and fit it to the data
#clf = tree.DecisionTreeClassifier()
#clf.fit(X, y)

## Use the classifier to predict new instances
#print(clf.predict([[0, 0], [0, 1], [1, 0], [1, 1]]))  # prints [0 1 1 0]


from sklearn import tree

# Define the dataset
X = [
 [1,0,0,1.521,0,0,  1,2,1,1.564,0,0],   #Helisova-Dlask
 [0,0,1,1.395,0,0,  1,0,1,1.584,0,1],   #Patho-Skopkova
 [0,0,1,1.718,0,0,  1,1,1,1.773,0,0],   #Priklopil-Koci
 #[1,0,0,1.521,0,0,  1,0,1,1.644,0,0],   #Helisova-Gvichiani
 [0,0,1,1.395,0,0,  1,2,1,1.564,0,0],   #Patho-Dlask
 [0,0,1,1.718,0,0,  1,0,1,1.584,0,1],   #Prikopil-Skopkova
 [1,1,1,1.419,0,1,  1,1,1,1.773,0,0],   #Jares-Koci
 [0,0,1,1.395,0,0,  1,0,1,1.644,0,0],   #Patho-Gvichiani
 [0,0,1,1.718,0,0,  1,2,1,1.564,0,0],   #Priklopil-Dlask
 [1,1,1,1.419,0,1,  1,0,1,1.584,0,1],   #Jares-Skopkova

 [1,1,1,1.419,0,1,  1,0,1,1.644,0,0],    #Jares-Gvichiani
 [0,0,1,1.718,0,0,  1,0,1,1.644,0,0],    #Priklopil-Gvichiani
 [1,0,0,1.521,0,0,  1,0,1,1.584,0,1]     #Helisova-Skopkova
]

y = [2,-2,-3,-3,3,-3,-3,-1,-2,-3,-3,-1]



#X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # input features
#y = [0, 1, 1, 0]  # corresponding output labels

# Create the classifier and fit it to the data
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Use the classifier to predict new instances
print(clf.predict([[1,0,0,1.521,0,0,  1,0,1,1.644,0,0]]))  # prints [0 1 1 0]