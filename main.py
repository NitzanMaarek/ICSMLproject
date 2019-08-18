import pandas as pd
# import pandas_profiling
import pygtrie
from typing import Tuple
from trie import Trie
import time
import random
import pickle


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

    # *** Read the dataset to memory ***
    # malicious_dataset = pd.read_parquet(r"data.parquet", engine='pyarrow', columns=["url"])
    # malicious_dataset = pd.read_csv(r"failed_dataset.csv")
    malicious_dataset = pd.read_csv(r"Malicious_urls_only_800k.csv")

    benign_dataset = pd.read_csv(r"alexa_top-1m.csv")
    # benign_dataset = benign_dataset.head(800000)

    # *** Initiating tries ***
    malicious_trie = Trie('Malicious')
    benign_trie = Trie('Benign_1m_Trie')

    # *** Creating malicious trie tree ***
    # mal_tree_time_start = time.time()
    # print('Started building malicious trie')
    # malicious_trie = insert_values_to_trie(malicious_trie, malicious_dataset)
    # mal_tree_time_end = time.time()
    # print('Finished building malicious trie in: ' + str(mal_tree_time_end - mal_tree_time_start))

    # # *** Creating benign trie tree ***
    # benign_tree_time_start = time.time()
    # print('Started building benign trie')
    # benign_trie = insert_values_to_trie(benign_trie, benign_dataset)
    # benign_tree_time_end = time.time()
    # print('Finished building benign trie in:     ' + str(benign_tree_time_end - benign_tree_time_start))
    #
    # # *** How deep down the tree did the malicious dataset get until first error ***
    # # url_dict = how_deep_did_malicious_dataset_get(malicious_dataset, benign_trie)
    # # print(url_dict)
    #
    #
    # # *** PROBABILITY TESTING *****
    # # some_node = malicious_trie.get_node_by_prefix(malicious_trie.get_root(), 'nseralum')
    # # a_node = malicious_trie.get_node_by_prefix(malicious_trie.get_root(), 'nseralum.')
    # # print(some_node.get_node_sequence_probability())
    # # print(a_node.get_node_sequence_probability())
    # # print('{0:.10f}'.format(some_node.get_node_sequence_probability()))
    # # print('{0:.10f}'.format(a_node.get_node_sequence_probability()))
    #
    #
    # # ***** PICKLE TESTING *****
    # # malicious_saving_time_start = time.time()
    # # save_trie_pickle(malicious_trie)
    # # malicious_saving_time_end = time.time()
    # # print('Finished saving malicious compressed trie structure in: ' + str(malicious_saving_time_end - malicious_saving_time_start))
    #
    # benign_saving_time_start = time.time()
    # save_trie_pickle(benign_trie)
    # benign_saving_time_end = time.time()
    # print('Finished saving benign compressed trie structure in: ' + str(benign_saving_time_end - benign_saving_time_start))
    # print('Program Finished')
    # time.sleep(4)
    # mal_trie_loaded = load_trie_pickle('Malicious.pickle')

    benign_loading_time_start = time.time()
    benign_trie_loaded = load_trie_pickle('Benign_1m_Trie.pickle')
    benign_loading_time_end = time.time()
    print('Finished loading benign compressed trie structure in: ' + str(benign_loading_time_end - benign_loading_time_start))

    # print(mal_trie_loaded)
    # print(benign_trie_loaded)


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
    with open(path, 'rb') as file:
        trie = pickle.load(file)
    # pickle.load(open(path, 'rb'))
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
        url_name = row[1]['url']
        root = benign_root
        calculate_random_deep_search(url_dict, url_name, benign_tree, root)

    return url_dict


def calculate_random_deep_search(url_dict, url_name, benign_tree, root):
    """

    :param url_dict:
    :param url_name:
    :param benign_tree:
    :param root:
    :return:
    """
    # benign_root = benign_tree.get_root
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

