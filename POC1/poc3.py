from sklearn import tree

# function to run the whole POC
def runPOC(tr_input:str, tr_output:str, tst_input:str, tst_output:str):
    # function for loading data from file to 'list of lists'
    # it expects that one row = one learning case, features are separated by ','
    def readMoreLines(fileName:str):
        output_list = []
        with open(fileName) as inputFile:
            for line in inputFile:
                line_splitted = line.rstrip().split(',')
                output_list.append(line_splitted)
        return output_list

    # function for loadig data from file to list
    # it expects that the file contains only one row of output labels separated by ','
    def readOneLine(fileName:str):
        with open(fileName) as output_file:
            line = output_file.readline()
            output_list = line.rstrip().split(',')
        return output_list

    # load input data for training
    X = readMoreLines(tr_input)

    # load output data for training
    y = readOneLine(tr_output)

    ##################################################
    # Create the classifier and fit it to the data
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    ##################################################

    # load input data for testing 
    Z = readMoreLines(tst_input)
    RC = readOneLine(tst_output)

    # Use the classifier to predict new instances
    result = clf.predict(Z)

    # simple comparison of expected and real results
    print('Expected result: ', RC)
    print('Predicted result: ', result)
    print('Success rate: ', len([x for x in RC == result if x == True])/ len(RC)*100, '%')


#runPOC('./POC1/inputs_outputs/input1.txt', './POC1/inputs_outputs/output1.txt', './POC1/inputs_outputs/testInput1.txt', './POC1/inputs_outputs/testOutput1.txt')
