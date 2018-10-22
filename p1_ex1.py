from table import *
from utils import *


def main():
    print("-------- Exercise 1 ---------")
    print("- Using the tf-idf approach to summarize an english textual document")
    print("<<< Summary of data/ex1.txt is >>>")
    directory = 'data/'
    tab = Table(['ex1.txt'], parse_sentences=True, language='english', path=directory)
    with open(directory+'ex1.txt', 'r', encoding='latin1') as file:
        doc = file.read()
        result = summary(doc, tab, 3)
        for sentence in result:
            print(sentence)


if __name__ == '__main__':
    main()
