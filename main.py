import csv

import pandas as pd
# import pandas_profiling
import pygtrie
from typing import Tuple
from trie import Trie
import time
import random
import pickle
import ast
import numpy as np


def create_pygtrie_of_dataset(df):
    """
    Creates dataset with column "url", creates and returns CharTrie from that column.
    :param df: Dataframe with 'url' column
    :return: CharTrie from 'url' column
    """
    trie = pygtrie.CharTrie()
    for url in df:
        url_len = len(url)
        for i in range(0, url_len):
            char = url[i]
            if i == url_len-1:
                trie[char] = True
            else:
                trie[char] = False
    return trie


def pipeline():

    print('The code here is not ready to run without reading the entire functionality of the project.'
          ' Thanks for consideration')

    # calculate_threshold_tpr_fpr()
    # *** Read the dataset to memory ***
    # malicious_dataset = pd.read_parquet(r"data.parquet", engine='pyarrow', columns=["url"])
    # malicious_dataset = pd.read_csv(r"failed_dataset.csv")

    # url_probability_dict_to_csv()
    # calculate_average_from_csv('Benign_100k_Test_Probability_results.csv', 'Benign_100k_Test_AVG_Prob_results.csv')
    # calculate_average_from_csv('Malicious_entire_dataset_prob_results.csv', 'Malicious_entire_dataset_AVG_Prob_results.csv')
    #



    # benign_dataset = pd.read_csv(r"alexa_top-1m.csv")
    # benign_dataset.sample(frac=1)
    # msk = np.random.rand(len(benign_dataset)) < 0.9
    # train_benign = benign_dataset[msk]
    # test_benign = benign_dataset[~msk]
    # test_benign.to_csv('Benign_100k_Test_Set.csv')
    #
    #
    # *** Initiating tries ***
    # malicious_trie = Trie('Malicious')
    # benign_trie = Trie('Benign_900k_Trie_Training_Set')


    # print(url_dict)


    # *** Creating malicious trie tree ***
    # mal_tree_time_start = time.time()
    # print('Started building malicious trie')
    # malicious_trie = insert_values_to_trie(malicious_trie, malicious_dataset)
    # mal_tree_time_end = time.time()
    # print('Finished building malicious trie in: ' + str(mal_tree_time_end - mal_tree_time_start))

    # *** Creating benign trie tree ***
    # benign_tree_time_start = time.time()
    # print('Started building benign trie')
    # benign_trie = insert_values_to_trie(benign_trie, train_benign)
    # benign_tree_time_end = time.time()
    # print('Finished building benign trie in:     ' + str(benign_tree_time_end - benign_tree_time_start))

    # *** How deep down the tree did the malicious dataset get until first error ***
    # url_dict = how_deep_did_malicious_dataset_get(test_benign, benign_trie)
    # print(url_dict)


    # *** PROBABILITY TESTING *****
    # some_node = malicious_trie.get_node_by_prefix(malicious_trie.get_root(), 'nseralum')
    # a_node = malicious_trie.get_node_by_prefix(malicious_trie.get_root(), 'nseralum.')
    # print(some_node.get_node_sequence_probability())
    # print(a_node.get_node_sequence_probability())
    # print('{0:.10f}'.format(some_node.get_node_sequence_probability()))
    # print('{0:.10f}'.format(a_node.get_node_sequence_probability()))


    # ***** PICKLE TESTING *****
    # malicious_saving_time_start = time.time()
    # save_trie_pickle(malicious_trie)
    # malicious_saving_time_end = time.time()
    # print('Finished saving malicious compressed trie structure in: ' + str(malicious_saving_time_end - malicious_saving_time_start))

    # benign_saving_time_start = time.time()
    # save_trie_pickle(benign_trie)
    # benign_saving_time_end = time.time()
    # print('Finished saving benign compressed trie structure in: ' + str(benign_saving_time_end - benign_saving_time_start))
    # print('Program Finished')
    # time.sleep(4)
    # mal_trie_loaded = load_trie_pickle('Malicious.pickle')

    # benign_loading_time_start = time.time()
    # benign_trie_loaded = load_trie_pickle('Benign_1m_Trie.pickle')
    # benign_loading_time_end = time.time()
    # print('Finished loading benign compressed trie structure in: ' + str(benign_loading_time_end - benign_loading_time_start))

    # print(mal_trie_loaded)
    # print(benign_trie_loaded)


def calculate_threshold_tpr_fpr():
    """
    Method prints results for different probability threshold for classification.
    """
    # benign_avg_prob_df = pd.read_csv('Benign_100k_Test_AVG_Prob_results.csv')
    # malicious_avg_prob_df = pd.read_csv('Malicious_entire_dataset_AVG_Prob_results.csv')

    # print('***** Calculating Benign Classification Results *****')
    # threshold_01 = 0
    # threshold_011 = 0
    # threshold_012 = 0
    # threshold_013 = 0
    # threshold_014 = 0
    # threshold_015 = 0
    # threshold_017 = 0
    # threshold_018 = 0
    # threshold_02 = 0
    # threshold_019 = 0
    # threshold_009 = 0
    # threshold_008 = 0
    # threshold_007 = 0
    # threshold_006 = 0
    # threshold_005 = 0
    # threshold_004 = 0
    # threshold_003 = 0
    # threshold_002 = 0
    # threshold_001 = 0
    #
    # for row in benign_avg_prob_df.iterrows():
    #
    #     prob = row[1]['Average Probability']

        # if prob > 0.1:
        #     threshold_01 += 1
        # if prob > 0.11:
        #     threshold_011 += 1
        # if prob > 0.12:
        #     threshold_012 += 1
        # if prob > 0.09:
        #     threshold_009 += 1
        # if prob > 0.08:
        #     threshold_008 += 1
        # if prob > 0.07:
        #     threshold_007 += 1
        # if prob > 0.13:
        #     threshold_013 += 1
        # if prob > 0.14:
        #     threshold_014 += 1
        # if prob > 0.15:
        #     threshold_015 += 1
        # if prob > 0.17:
        #     threshold_017 += 1
        # if prob > 0.18:
        #     threshold_018 += 1
        # if prob > 0.19:
        #     threshold_019 += 1
        # if prob > 0.2:
        #     threshold_02 += 1
        # if prob > 0.06:
        #     threshold_006 += 1
        # if prob > 0.05:
        #     threshold_005 += 1
        # if prob > 0.04:
        #     threshold_004 += 1
        # if prob > 0.03:
        #     threshold_003 += 1
        # if prob > 0.02:
        #     threshold_002 += 1
        # if prob > 0.01:
        #     threshold_001 += 1

    # print('Threshold 0.1: Accuracy of ' + str(threshold_01) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_01/len(benign_avg_prob_df)))
    # print('Threshold 0.11: Accuracy of ' + str(threshold_011) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_011 / len(benign_avg_prob_df)))
    # print('Threshold 0.12: Accuracy of ' + str(threshold_012) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_012 / len(benign_avg_prob_df)))
    # print('Threshold 0.09: Accuracy of ' + str(threshold_009) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_009 / len(benign_avg_prob_df)))
    # print('Threshold 0.08: Accuracy of ' + str(threshold_008) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_008 / len(benign_avg_prob_df)))
    # print('Threshold 0.07: Accuracy of ' + str(threshold_007) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_007 / len(benign_avg_prob_df)))
    # print('Threshold 0.13: Accuracy of ' + str(threshold_013) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_013 / len(benign_avg_prob_df)))
    # print('Threshold 0.14: Accuracy of ' + str(threshold_014) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_014 / len(benign_avg_prob_df)))
    # print('Threshold 0.15: Accuracy of ' + str(threshold_015) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_015 / len(benign_avg_prob_df)))
    # print('Threshold 0.17: Accuracy of ' + str(threshold_017) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_017 / len(benign_avg_prob_df)))
    # print('Threshold 0.18: Accuracy of ' + str(threshold_018) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_018 / len(benign_avg_prob_df)))
    # print('Threshold 0.19: Accuracy of ' + str(threshold_019) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_019 / len(benign_avg_prob_df)))
    # print('Threshold 0.2: Accuracy of ' + str(threshold_02) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_02 / len(benign_avg_prob_df)))
    # print('Threshold 0.06: Accuracy of ' + str(threshold_006) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_006 / len(benign_avg_prob_df)))
    # print('Threshold 0.05: Accuracy of ' + str(threshold_005) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_005 / len(benign_avg_prob_df)))
    # print('Threshold 0.04: Accuracy of ' + str(threshold_004) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_004 / len(benign_avg_prob_df)))
    # print('Threshold 0.03: Accuracy of ' + str(threshold_003) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_003 / len(benign_avg_prob_df)))
    # print('Threshold 0.02: Accuracy of ' + str(threshold_002) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_002 / len(benign_avg_prob_df)))
    # print('Threshold 0.01: Accuracy of ' + str(threshold_001) + ' out of: ' + str(len(benign_avg_prob_df)) + ' is: '
    #       + str(threshold_001 / len(benign_avg_prob_df)))

    # print('***** Calculating Malicious Classification Results *****')
    # threshold_01 = 0
    # threshold_011 = 0
    # threshold_012 = 0
    # threshold_009 = 0
    # threshold_008 = 0
    # threshold_007 = 0
    # threshold_013 = 0
    # threshold_014 = 0
    # threshold_015 = 0
    # threshold_017 = 0
    # threshold_018 = 0
    # threshold_02 = 0
    # threshold_019 = 0
    # threshold_006 = 0
    # threshold_005 = 0
    # threshold_004 = 0
    # threshold_003 = 0
    # threshold_002 = 0
    # threshold_001 = 0
    #
    # for row in malicious_avg_prob_df.iterrows():
    #
    #     prob = row[1]['Average Probability']

        # if prob > 0.1:
        #     threshold_01 += 1
        # if prob > 0.11:
        #     threshold_011 += 1
        # if prob > 0.12:
        #     threshold_012 += 1
        # if prob > 0.09:
        #     threshold_009 += 1
        # if prob > 0.08:
        #     threshold_008 += 1
        # if prob > 0.07:
        #     threshold_007 += 1
        # if prob > 0.13:
        #     threshold_013 += 1
        # if prob > 0.14:
        #     threshold_014 += 1
        # if prob > 0.15:
        #     threshold_015 += 1
        # if prob > 0.17:
        #     threshold_017 += 1
        # if prob > 0.18:
        #     threshold_018 += 1
        # if prob > 0.19:
        #     threshold_019 += 1
        # if prob > 0.2:
        #     threshold_02 += 1
        # if prob > 0.06:
        #     threshold_006 += 1
        # if prob > 0.05:
        #     threshold_005 += 1
        # if prob > 0.04:
        #     threshold_004 += 1
        # if prob > 0.03:
        #     threshold_003 += 1
        # if prob > 0.02:
        #     threshold_002 += 1
        # if prob > 0.01:
        #     threshold_001 += 1

    # print('Threshold 0.1: Accuracy of ' + str(threshold_01) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_01 / len(malicious_avg_prob_df)))
    # print('Threshold 0.11: Accuracy of ' + str(threshold_011) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_011 / len(malicious_avg_prob_df)))
    # print('Threshold 0.12: Accuracy of ' + str(threshold_012) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_012 / len(malicious_avg_prob_df)))
    # print('Threshold 0.09: Accuracy of ' + str(threshold_009) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_009 / len(malicious_avg_prob_df)))
    # print('Threshold 0.08: Accuracy of ' + str(threshold_008) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_008 / len(malicious_avg_prob_df)))
    # print('Threshold 0.07: Accuracy of ' + str(threshold_007) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_007 / len(malicious_avg_prob_df)))
    # print('Threshold 0.13: Accuracy of ' + str(threshold_013) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_013 / len(malicious_avg_prob_df)))
    # print('Threshold 0.14: Accuracy of ' + str(threshold_014) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_014 / len(malicious_avg_prob_df)))
    # print('Threshold 0.15: Accuracy of ' + str(threshold_015) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_015 / len(malicious_avg_prob_df)))
    # print('Threshold 0.17: Accuracy of ' + str(threshold_017) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_017 / len(malicious_avg_prob_df)))
    # print('Threshold 0.18: Accuracy of ' + str(threshold_018) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_018 / len(malicious_avg_prob_df)))
    # print('Threshold 0.19: Accuracy of ' + str(threshold_019) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_019 / len(malicious_avg_prob_df)))
    # print('Threshold 0.2: Accuracy of ' + str(threshold_02) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_02 / len(malicious_avg_prob_df)))
    # print('Threshold 0.06: Accuracy of ' + str(threshold_006) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_006 / len(malicious_avg_prob_df)))
    # print('Threshold 0.05: Accuracy of ' + str(threshold_005) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_005 / len(malicious_avg_prob_df)))
    # print('Threshold 0.04: Accuracy of ' + str(threshold_004) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_004 / len(malicious_avg_prob_df)))
    # print('Threshold 0.03: Accuracy of ' + str(threshold_003) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_003 / len(malicious_avg_prob_df)))
    # print('Threshold 0.02: Accuracy of ' + str(threshold_002) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_002 / len(malicious_avg_prob_df)))
    # print('Threshold 0.01: Accuracy of ' + str(threshold_001) + ' out of: ' + str(len(malicious_avg_prob_df)) + ' is: '
    #       + str(threshold_001 / len(malicious_avg_prob_df)))


def url_probability_dict_to_csv():
    """
    Run probability tests on different csv sizes of malicious dataset.
    Save result dictionary in csv files.
    """

    # benign_trie_loaded = load_trie_pickle('Benign_1m_Trie.pickle')
    benign_trie_loaded = load_trie_pickle('Benign_900k_Trie_Training_Set.pickle')

    # malicious_dataset_100 = pd.read_csv(r"Malicious_urls_only_100.csv")
    # malicious_dataset_10k = pd.read_csv(r"Malicious_urls_only_10k.csv")
    # malicious_dataset_100k = pd.read_csv(r"Malicious_urls_only_100k.csv")
    # malicious_dataset_300k = pd.read_csv(r"Malicious_urls_only_300k.csv")
    # malicious_dataset_500k = pd.read_csv(r"Malicious_urls_only_500k.csv")
    # malicious_dataset_800k = pd.read_csv(r"Malicious_urls_only_800k.csv")
    # malicious_dataset_1m = pd.read_csv(r"Malicious_urls_only_1m.csv")
    malicious_dataset_all = pd.read_csv(r"Malicious_Urls_Entire_dataset.csv")

    # benign_dataset_test_100k = pd.read_csv(r"Benign_100k_Test_Set.csv")

    # print('*** 100k Benign instances ***')
    # start = time.time()
    # url_dict = how_deep_did_malicious_dataset_get(benign_dataset_test_100k, benign_trie_loaded)
    # end = time.time()
    # print('Process took: ' + str(end - start))
    # save_dict_as_df('Benign_100k_Test_Probability_results', url_dict)
    # print(url_dict)
    #
    # print('*** 100 Malicious Instances ***')
    # start = time.time()
    # url_dict = how_deep_did_malicious_dataset_get(malicious_dataset_100, benign_trie_loaded)
    # end = time.time()
    # print('Process took: ' + str(end - start))
    # save_dict_as_df('Malicious_100_prob_results', url_dict)
    # # print(url_dict)
    #
    # print('*** 10k Malicious Instances ***')
    # start = time.time()
    # url_dict = how_deep_did_malicious_dataset_get(malicious_dataset_10k, benign_trie_loaded)
    # end = time.time()
    # print('Process took: ' + str(end - start))
    # save_dict_as_df('Malicious_10k_prob_results', url_dict)
    #
    # # print(url_dict)
    #
    # print('*** 100k Malicious Instances ***')
    # start = time.time()
    # url_dict = how_deep_did_malicious_dataset_get(malicious_dataset_100k, benign_trie_loaded)
    # end = time.time()
    # print('Process took: ' + str(end - start))
    # save_dict_as_df('Malicious_100k_prob_results', url_dict)
    # # print(url_dict)

    # print('*** 300k Malicious Instances ***')
    # start = time.time()
    # url_dict = how_deep_did_malicious_dataset_get(malicious_dataset_300k, benign_trie_loaded)
    # end = time.time()
    # print('Process took: ' + str(end - start))
    # save_dict_as_df('Malicious_300k_prob_results', url_dict)
    # # print(url_dict)
    #
    # print('*** 500k Malicious Instances ***')
    # start = time.time()
    # url_dict = how_deep_did_malicious_dataset_get(malicious_dataset_500k, benign_trie_loaded)
    # end = time.time()
    # print('Process took: ' + str(end - start))
    # save_dict_as_df('Malicious_500k_prob_results', url_dict)
    # # print(url_dict)
    #
    # print('*** 800k Malicious Instances ***')
    # start = time.time()
    # url_dict = how_deep_did_malicious_dataset_get(malicious_dataset_800k, benign_trie_loaded)
    # end = time.time()
    # print('Process took: ' + str(end - start))
    # save_dict_as_df('Malicious_800k_prob_results', url_dict)
    # # print(url_dict)
    #
    # print('*** 1m Malicious Instances ***')
    # start = time.time()
    # url_dict = how_deep_did_malicious_dataset_get(malicious_dataset_1m, benign_trie_loaded)
    # end = time.time()
    # print('Process took: ' + str(end - start))
    # save_dict_as_df('Malicious_1m_p

    print('*** All Malicious Instances ***')
    start = time.time()
    url_dict = how_deep_did_malicious_dataset_get(malicious_dataset_all, benign_trie_loaded)
    end = time.time()
    print('Process took: ' + str(end - start))
    save_dict_as_df('Malicious_entire_dataset_prob_results', url_dict)


def calculate_average_from_csv(file_name, result_file_name):
    """
    Calculate average of probabilities for each URL Name
    :param file_name: CSV file with columns: 'URL Name' and 'Probabilities'
    :param result_file_name: CSV file name to save results in
    """
    df = pd.read_csv(file_name)
    url_name_to_avg_prob_df = []
    url_name_to_avg_prob_dict = {}
    counter = 0
    print('Calculating average probabilities')
    for row in df.iterrows():
        prob_arr = row[1]['Probabilities']
        prob_arr = ast.literal_eval(prob_arr)
        url_name = row[1]['URL Name']
        prob_arr_len = len(prob_arr)
        if prob_arr_len == 0:
            continue
        sum_probabilities = 0.0
        for prob in prob_arr:
            # if prob == 1:
            #     # print(url_name)
            #     prob = 0
            sum_probabilities = sum_probabilities + prob
        avg_probabilities = sum_probabilities/prob_arr_len
        if avg_probabilities > 0.09:
            counter += 1
        url_name_to_avg_prob_dict[url_name] = avg_probabilities
    url_name_to_avg_prob_df = pd.DataFrame(list(url_name_to_avg_prob_dict.items()), columns=['URL Name', 'Average Probability'])
    url_name_to_avg_prob_df.to_csv(result_file_name)
    print('Number of URLs with probability over 0.09 is: ' + str(counter))
    print('Accuracy of: ' + str(counter) + ' out of: ' + str(len(df)) + ' is: ' + str(counter / len(df)))


def save_dict_as_df(file_name, given_dict):
    """
    Given dictionary (key = URL name, value = Probability array), save data to csv.
    :param file_name: Name of csv file
    :param dict: Dictionary
    """
    print('Saving dictionary as CSV.')
    print(type(given_dict))
    csv_columns = ['URL Name', 'Probabilities']
    df = pd.DataFrame(list(given_dict.items()), columns=csv_columns)
    df.to_csv(file_name + '.csv')
    print('Finished saving dictionary as CSV.')


def save_trie_pickle(trie):
    with open(trie.description + '.pickle', 'wb') as file:
        pickle.dump(trie, file, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dumps(trie, open(trie.description + '.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_trie_pickle(path):
    """
    Loads Trie object from pickle file
    :param path: path of file
    :return: Trie object
    """
    print('Started loading trie ' + path)
    start = time.time()
    with open(path, 'rb') as file:
        trie = pickle.load(file)
    # pickle.load(open(path, 'rb'))
    end = time.time()
    print('Finished loading trie successfully in ' + str(end - start) + ' seconds.')
    return trie


def how_deep_did_malicious_dataset_get(mal_df, benign_tree: Trie):
    """
    We check how deep down the benign trie each domain name in the malicious dataset get
    :param mal_df: Malicious Pandas Dataframe with column 'url' for domain name
    :param benign_tree: Trie of benign domain names dataset
    :return: Dictionary(url_name, Array[]) - The array is the same len as the url_name.
             0 in a cell in the array means same char, 1 in a cell means char was not found (changed perhaps)
    """
    url_dict = {}
    benign_root = benign_tree.get_root()
    for row in mal_df.iterrows():
        url_name = row[1][1]
        root = benign_root
        calculate_random_probability_deep_search(url_dict, url_name, benign_tree, root)

    return url_dict


def calculate_random_deep_search(url_dict, url_name, benign_tree, root):
    """
    Method checks how deep down the Trie does the given url_name get without finding error.
    Once an error found (no child with letter), then select randomly a child and continue search from it.
    :param url_dict: Dictionary (Key = url_name, value = Array where 0 means found and 1 not found).
    :param url_name: Url to search down the Trie
    :param benign_tree: Benign Trie
    :param root: Root of Benign Trie
    """
    # benign_root = benign_tree.get_root
    url_len = len(url_name)
    url_dict[url_name] = [1] * url_len      # Init dictionary value with a list of 0's the size of the url string.
    lookup = ''
    # prev_suffix = ''
    last_correct_node = None
    search_word = url_name
    search_word_counter = 0
    for i in range(0, url_len - 1):
        char = search_word[search_word_counter]
        lookup += char
        search_result_node = benign_tree.get_node_by_prefix(root, lookup)  # return tuple = (true/false, node counter)
        if search_result_node is not None:  # prefix found
            url_dict[url_name][i] = 0
            last_correct_node = search_result_node
            search_word_counter += 1
        else:
            url_dict[url_name][i] = 1
            # last_correct_node = benign_tree.get_node_by_prefix(benign_root, prev_suffix)
            if last_correct_node is not None:
                if last_correct_node.word_finished:
                    mark_the_rest_1s(url_dict, url_name, i)
                    break
                    # TODO: maybe add the new domain to subtire here?
                else:
                    lookup = ''  # start looking from one of the children as "root" in subtrie
                    search_word_counter = 0
                    random_child = last_correct_node.children[random.randint(0, len(last_correct_node.children) - 1)]
                    root = random_child
                    search_word = url_name[i+1:]  # rest of string to search
                    # search_word = list(search_word)
                    # search_word[1] = str(random_child.char)
                    # search_word = ''.join(search_word)


def calculate_random_probability_deep_search(url_dict, url_name, benign_tree, root):
    """
    Method checks how deep down the Trie does the given url_name get without finding error.
    Once an error found (no child with letter), then select randomly a child and continue search from it.
    :param url_dict: Dictionary (Key = url_name, value = Array where 0 means not found, else is the probability of the node).
    :param url_name: Url to search down the Trie
    :param benign_tree: Benign Trie
    :param root: Root of Benign Trie
    """
    # benign_root = benign_tree.get_root
    try:
        url_len = len(url_name)
    except TypeError:
        print('This URL name was float type: ' + str(url_name))
        url_name = str(url_name)
        url_len = len(url_name)
    url_dict[url_name] = [0] * url_len      # Init dictionary value with a list of 0's the size of the url string.
    lookup = ''
    # prev_suffix = ''
    last_correct_node = None
    search_word = url_name
    search_word_counter = 0
    for i in range(0, url_len - 1):
        char = search_word[search_word_counter]
        lookup += char
        search_result_node = benign_tree.get_node_by_prefix(root, lookup)  # return tuple = (true/false, node counter)
        if type(search_result_node) is tuple:
            print('Why is this tuple?')
        if search_result_node is not None:  # prefix found

            url_dict[url_name][i] = search_result_node.get_node_probability()
            last_correct_node = search_result_node
            search_word_counter += 1
        else:
            url_dict[url_name][i] = 0       # probability of 0
            # last_correct_node = benign_tree.get_node_by_prefix(benign_root, prev_suffix)
            if last_correct_node is not None:
                if last_correct_node.word_finished:
                    mark_the_rest_0s(url_dict, url_name, i)
                    break
                    # TODO: maybe add the new domain to subtire here?
                else:
                    lookup = ''  # start looking from one of the children as "root" in subtrie
                    search_word_counter = 0
                    random_child = last_correct_node.children[random.randint(0, len(last_correct_node.children) - 1)]
                    root = random_child
                    search_word = url_name[i+1:]  # rest of string to search
                    # search_word = list(search_word)
                    # search_word[1] = str(random_child.char)
                    # search_word = ''.join(search_word)


def mark_the_rest_1s(url_dict, url_name, i):
    """

    :param url_dict:
    :param url_name:
    :param i:
    :return:
    """
    for j in range(i+1, len(url_name)-1):
        url_dict[url_name][j] = 1
    # return url_dict


def mark_the_rest_0s(url_dict, url_name, i):
    """

    :param url_dict:
    :param url_name:
    :param i:
    :return:
    """
    for j in range(i+1, len(url_name)-1):
        url_dict[url_name][j] = 0
    # return url_dict

def insert_values_to_trie(trie, df):
    """
    Inserts strings from given df into given trie object
    :param trie: Trie
    :param df: Pandas dataframe
    :return: Trie
    """
    trie_root = trie.get_root()
    i = 0
    for row in df.iterrows():
        word = row[1]['url']
        if i % 10000 == 0:
            print(i)
            print(word)
        i += 1
        trie.add(trie_root, word)

    return trie


if __name__ == "__main__":
    pipeline()
    print('Out of Pipeline')

    # malicious_urls = create_pygtrie_of_dataset(malicious_dataset.head(100))
    # benign_trie = create_pygtrie_of_dataset(benign_dataset)
    # print("Malicious Trie length (number of values) is: " + str(malicious_urls.__len__()))
    # print("Benign Trie length (number of values) is: " + str(benign_trie.__len__()))


    # root = TrieNode('*')
    # for url_name in url_names:
    #     add(root, url_name)
    # add(root, "hackathon")
    # add(root, 'hack')
    # print(root)
    # print(find_prefix(root, 'hac'))
    # print(find_prefix(root, 'hack'))
    # print(find_prefix(root, 'hackathon'))
    # print(find_prefix(root, 'ha'))
    # print(find_prefix(root, 'hammer'))

    # df.to_csv('data.csv')  #DONT RUN THIS ITLL TAKE LOTS TIME AND MEMORY

