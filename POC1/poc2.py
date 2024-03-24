from sklearn import tree

# function for loading data from file to 'list of lists'
# it expects that one row = one learning case, features are separated by ','
def readMoreLines(fileName:str):
    output_list = []
    with open(fileName) as inputFile:
        for line in inputFile:
            line_splitted = line.rstrip().split(',')
            output_list.append(line_splitted)
    return output_list   # [[1,2],[3],[2,2,9]]

#TODO reimplement readMoreLines so it uses readOneLine

def readOneLine(fileName:str):
    with open(fileName) as output_file:
        line = output_file.readline()
        output_list = line.rstrip().split(',')
    return output_list #[-1,-2,1]

# load input data for training
X = readMoreLines('./POC1/inputs_outputs/input1.txt')

# load output data for training
y = readOneLine('./POC1/inputs_outputs/output1.txt')


#TODO check if X and y have the same length

##################################################
# Create the classifier and fit it to the data
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
##################################################

# load input data for testing 
Z = readMoreLines('./POC1/inputs_outputs/testInput1.txt')
RC = readOneLine('./POC1/inputs_outputs/testOutput1.txt')

# Use the classifier to predict new instances
result = clf.predict(Z)

# simple comparison of expected and real results
print(result == RC)

