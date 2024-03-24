from sklearn import tree

# Define the dataset for training - each list is pair of player who played a match
X = [
 [1,0,0,1.521,0,0,  1,2,1,1.564,0,0],   #Helisova-Dlask 3:1
 [0,0,1,1.395,0,0,  1,0,1,1.584,0,1],   #Patho-Skopkova 1:3
 [0,0,1,1.718,0,0,  1,1,1,1.773,0,0],   #Priklopil-Koci
 #[1,0,0,1.521,0,0,  1,0,1,1.644,0,0],   #Helisova-Gvichiani  - not included in the training data set - will be predicted
 [0,0,1,1.395,0,0,  1,2,1,1.564,0,0],   #Patho-Dlask
 [0,0,1,1.718,0,0,  1,0,1,1.584,0,1],   #Prikopil-Skopkova
 [1,1,1,1.419,0,1,  1,1,1,1.773,0,0],   #Jares-Koci
 [0,0,1,1.395,0,0,  1,0,1,1.644,0,0],   #Patho-Gvichiani
 [0,0,1,1.718,0,0,  1,2,1,1.564,0,0],   #Priklopil-Dlask
 [1,1,1,1.419,0,1,  1,0,1,1.584,0,1],   #Jares-Skopkova

 [1,1,1,1.419,0,1,  1,0,1,1.644,0,0],    #Jares-Gvichiani      - using predicted result
 [0,0,1,1.718,0,0,  1,0,1,1.644,0,0],    #Priklopil-Gvichiani  - using predicted result
 [1,0,0,1.521,0,0,  1,0,1,1.584,0,1]     #Helisova-Skopkova
]

y = [2,-2,-3,-3,3,-3,-3,-1,-2,-3,-3,-1]   # training dataset output labels = match results (e.g. 3:2 means 3-2=1)


# Create the classifier and fit it to the data
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Use the classifier to predict new instances
print(clf.predict([[1,0,0,1.521,0,0,  1,0,1,1.644,0,0]]))  # predicts match Helisova-Gvichiani