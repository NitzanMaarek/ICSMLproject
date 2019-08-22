from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
#conclusions
#read data in chunks: better to seperate the data to some smaller parquets
def main():
	print(levenshtein_ratio_and_distance('google','bazazagoogle'))
	print('------------\n------------\nonly stide\n------------\n------------\n')
	test('with_added_features_fullk2.csv',False)
	print('------------\n------------\neverything unleashed\n------------\n------------\n')
	#test('fullfeatures.parquet',True)
	#generate_word_dictionary()
	# y = pd.read_parquet('data.parquet', engine='pyarrow')
	# parquet_file = pq.ParquetFile('data.parquet')
	# print(parquet_file.metadata)
	#benign_words = generate_word_dictionary()
	#windows,number_of_windows = get_windows_vector(benign_words)
	#add_features(benign_words,windows,number_of_windows) #check also is substring

def test(path,is_parquet):
	seed = 96
	test_size = 0.2
	df = pd.read_csv(path) if not is_parquet else pq.ParquetFile('data.parquet').read_row_group(0).to_pandas()
	X, Y = df.loc[:,(df.columns!='label') & (df.columns!='url')],df.loc[:,['label']]
	X = X.iloc[:,1:]
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
	# fit model no training data
	model = XGBClassifier()
	model.fit(X_train, y_train.values.ravel(),verbose=True)
	# make predictions for test data
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]
	# evaluate predictions
	accuracy = accuracy_score(y_test, predictions)
	matrix = confusion_matrix(y_test, predictions)
	plot_roc(y_test, model.predict_proba(X_test))
	print('confusion matrix \n')
	print(matrix)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

def plot_roc(y_test, predictions):
	predictions = predictions[:, 1]
	# calculate roc curve
	fpr, tpr, thresholds = roc_curve(y_test, predictions)
	# plot no skill
	pyplot.plot([0, 1], [0, 1], linestyle='--')
	# plot the roc curve for the model
	pyplot.plot(fpr, tpr)
	# show the plot
	pyplot.show()
	print ('AUC:')
	print(auc(fpr,tpr))

def generate_word_dictionary():
	data = pd.read_csv('top.csv',header=None,usecols=[1])
	all_words = {'name':[]}
	all_words_so_far = {}
	i = 0
	for j in data.iterrows(): #iterate over all the domain names 
		name = j[1][1]
		add_names_to_dict(all_words_so_far,name)
		# print(name)
		i+=1 
		if(i%10000==0):
			print('{}%'.format(i/10000)) 
	all_words['name'] = list(all_words_so_far.keys())
	result = pd.DataFrame.from_dict(all_words)
	print('nicer')
	result.to_csv('nice2.csv')
	return result
        

def add_names_to_dict(all_words_so_far,name):
	name = name.replace('-','.')
	splat_list = name.split('.')
	for name in splat_list:
		if name not in all_words_so_far:
			all_words_so_far[name] = 1
	#map an expander over the splat list
	#all_words_so_far['name']= Union(name.split('.'),current_list)

	
def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """

    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])

if __name__ == "__main__":
    main()