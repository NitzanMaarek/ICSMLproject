k=4 #should be fine, need to also try 4 but we'll see
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def get_all_k_windows(urls):
#gets all the benign urls and creates a vector sized a maximum of 36^k (for all the possible k-grams)
    windows_number = 0 
    result = {}
    for url in urls:
        s = url.lower()
        for i in range(len(s)-k+1):
            windows_number+=1
            token = url[i:i+k]
            if token not in result:
                result[token] = 1
            else:
                result[token] += 1
    #print(result)
    return result, windows_number


def add_current_k_windows(windows_dict,current_windows_number,url):
#gets all the benign urls and creates a vector sized a maximum of 36^k (for all the possible k-grams)
    windows_number = current_windows_number 
    s = url.lower()
    for i in range(len(s)-k+1):
        windows_number+=1
        token = url[i:i+k]
        if token not in windows_dict:
            windows_dict[token] = 1
        else:
            windows_dict[token] += 1
    return windows_dict, windows_number
    
#compute the combined probability for a url based on the log-sum combination method
def get_combined_score(url,training_dict,windows_number):
    combined_score = 0
    count = 0
    probability = 1
    s = url.lower()
    if(len(s)<=k-1):
        return 0
    for i in range(len(s)-k+1):
        token = url[i:i+k]
        if token not in training_dict:
            probability=0
            #currently we'll take an 0 as likness of 1 squared, later we can add a threshold and do it in a binary way

            combined_score += np.log2(1/(windows_number^2))
        else:
            probability = training_dict[token]/windows_number #the probabilty to see the current window
            combined_score += np.log2(probability)
        count+=1
    combined_score /= count
    return combined_score 


    
a,b =get_all_k_windows(['nice','nice','veryberrynice','nicer','nicey','niceee'])
print( a)
print(b)
print('the first:{}\nthe second:{}\nthe third:{}'.format(get_combined_score("nice",a,b),get_combined_score("mahalta",a,b),get_combined_score("mahaltaice",a,b)))

a,b =get_all_k_windows(['w3schools','yahoo'])
print( a)
print(b)

def trying():
    data = pd.read_csv('nice2.csv')
    lst = list()
    for j in data['name'].values: #iterate over all the domain names 
        if(j == j): # check for not nan
            lst.append(j)
    a,b = get_all_k_windows(lst)  
    #print(b)
    print('the first:{}\nthe second:{}\nthe third:{}'.format(get_combined_score("google",a,b),get_combined_score("mahalta",a,b),get_combined_score("gfdgdfr",a,b)))

def add_features_to_dataset():
    parquet_file = pq.ParquetFile('data.parquet')
    df = parquet_file.read_row_group(0).to_pandas()
    for num in range(1,22):
        df = pd.concat([df,parquet_file.read_row_group(num).to_pandas()],ignore_index = True)
    normality_scores = get_url_normality_scores(df['url'].values)
    df['stide'] = normality_scores
    df.to_csv('with_added_features_fullk4.csv',columns=['url','label','stide'])
    #table = pa.Table.from_pandas(df)
    #pq.write_table(table,'fullfeatures.parquet')

def get_url_normality_scores(names):
    data = pd.read_csv('nice2.csv')
    lst = list()
    for j in data['name'].values: #iterate over all the domain names 
        if(j == j): # check for not nan
            lst.append(j)
    windows,count = get_all_k_windows(lst)
    scores = []
    for name in names:
        scores.append(get_combined_score(name,windows,count))
    return scores

#trying()
add_features_to_dataset()

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},{'a': 100, 'b': 200, 'c': 300, 'd': 400},{'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
print(df)
X, y = df.loc[:,(df.columns!='d') & (df.columns!='c')],df.loc[:,['d']]
print(X)
print(y)
