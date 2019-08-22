from typing import Tuple


class TrieNode(object):
    """
    Class of a Trie Node.
    A node contains its' char, accumulating chars, counter of occurrences, flag if word completed and pointer to parent.
    Root node has None as a parent pointer.
    """

    def __init__(self, char: str, acc_word, parent):
        self.char = char
        self.children = []
        # Is it the last character of the word.`
        self.word_finished = False
        # How many times this character appeared in the addition process
        self.counter = 1
        self.accumulative_value = acc_word
        self.parent = parent

    def get_node_sequence_probability(self):
        """
        Method returns for given node the (empirical) conditional probability
        :return: Float = the probability
        """
        if self.parent is not None:     # Node isn't root
            parent_probability = self.parent.get_node_sequence_probability()
            total_occurrences_under_parent = self.parent.get_number_of_children_occurrences()
        else:       # Node is root
            parent_probability = 1
            total_occurrences_under_parent = self.get_number_of_children_occurrences()
        node_probability = self.counter/total_occurrences_under_parent
        node_conditional_probability = node_probability * parent_probability
        return node_conditional_probability

    def get_node_probability(self):
        """
        Method returns the probability of node given its' parent.
        :return: Float = the probability
        """
        if self.parent is not None:     # Node isn't root
            total_occurrences_under_parent = self.parent.get_number_of_children_occurrences()
        else:   # Node is root
            total_occurrences_under_parent = self.get_number_of_children_occurrences()
        node_probability = self.counter/total_occurrences_under_parent
        return node_probability

    def get_number_of_children_occurrences(self):
        """
        Method sums all of the node's children's counters.
        :return: Total counter of all children.
        """
        number_of_children_occurrences = 0

        for child in self.children:
            number_of_children_occurrences += child.counter

        return number_of_children_occurrences


class Trie:

    def __init__(self, description):
        """
        Return root
        """
        self.description = description
        self.root = TrieNode('*', '', None)

    def get_root(self):
        return self.root

    def add(self, root, word: str):
        """
        Adding a word in the trie structure
        """
        node = root
        try:
            for char in word:
                found_in_child = False
                # Search for the character in the children of the present `node`
                for child in node.children:
                    if child.char == char:
                        # We found it, increase the counter by 1 to keep track that another
                        # word has it as well
                        child.counter += 1
                        # And point the node to the child that contains this char
                        node = child
                        found_in_child = True
                        break
                # We did not find it so add a new child
                if not found_in_child:
                    new_node = TrieNode(char, node.accumulative_value+char, parent=node)
                    node.children.append(new_node)
                    # And then point node to the new child
                    node = new_node
            # Everything finished. Mark it as the end of a word.
        except TypeError:
            print(word)
        node.word_finished = True

    def find_prefix(self, root, prefix: str) -> Tuple[bool, int]:
        """
        Check and return
          1. If the prefix exists in any of the words we added so far
          2. If yes then how may words actually have the prefix
        """
        node = root
        # If the root node has no children, then return False.
        # Because it means we are trying to search in an empty trie
        if not root.children:
            return False, 0
        for char in prefix:
            char_not_found = True
            # Search through all the children of the present `node`
            for child in node.children:
                if child.char == char:
                    # We found the char existing in the child.
                    char_not_found = False
                    # Assign node as the child containing the char and break
                    node = child
                    break
            # Return False anyway when we did not find a char.
            if char_not_found:
                return False, 0
        # Well, we are here means we have found the prefix. Return true to indicate that
        # And also the counter of the last node. This indicates how many words have this
        # prefix
        return True, node.counter

    def get_node_by_prefix(self, root, prefix: str):
        """
        Check and return
            1. If the prefix exists
            2. return node
            3. else return None
        """
        node = root
        # If the root node has no children, then return False.
        # Because it means we are trying to search in an empty trie
        if not root.children:
            return None
        for char in prefix:
            char_not_found = True
            # Search through all the children of the present `node`
            for child in node.children:
                if child.char == char:
                    # We found the char existing in the child.
                    char_not_found = False
                    # Assign node as the child containing the char and break
                    node = child
                    break
            # Return False anyway when we did not find a char.
            if char_not_found:
                return None
        # Well, we are here means we have found the prefix. Return true to indicate that
        # And also the counter of the last node. This indicates how many words have this
        # prefix
        return node
