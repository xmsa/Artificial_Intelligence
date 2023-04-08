import os
import numpy as np
import pandas as pd


class Node:
    def __init__(self, name, parent, tabel):
        self.__name = name
        self.__parent = parent
        self.tabel = tabel

    @property
    def name(self):
        return self.__name

    @property
    def parent(self):
        return self.__parent

    @property
    def tabel(self):
        return self.__tabel

    @tabel.setter
    def tabel(self, value):
        self.__tabel = pd.DataFrame(
            value, columns=self.parent + [self.name, 'prob'])


class Query:
    def __init__(self, query):
        self.__query = query


class Network:
    def __init__(self):
        self.nodes = list()

    def add_node(self, name, parent, tabel):
        node = Node(name, parent, tabel)
        self.nodes.append(node)

    @staticmethod
    def read_file(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File {filename} not found')

        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.strip(), lines))

        nw = Network()
        query = Query(lines.pop())
        size_network = int(lines.pop(0))
        for i in range(size_network):
            name = lines.pop(0)
            line = lines.pop(0)
            if is_float(line):
                parent = []
                value = float(line)
                tabel = np.array([[1, value], [0, 1-value]])
            else:
                parent = line.split(' ')
                tabel = []
                for _ in range(2**len(parent)):
                    line = lines.pop(0)
                    value = line.split(' ')
                    value = list(map(float, value))
                    tabel.append(value[:-1] + [1, value[-1]])
                    tabel.append(value[:-1] + [0, 1-value[-1]])
                tabel = np.array(tabel)
            nw.add_node(name, parent, tabel)

        return nw, query


def is_float(string):
    try:
        float(string)
        return True
    except:
        return False


if __name__ == '__main__':
    filename = 'input.txt'
    nw, query = Network.read_file(filename=filename)
