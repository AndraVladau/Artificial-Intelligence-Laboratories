"""
3. A file containing a text (consisting of several sentences) in Romanian is given - see the file "data/texts.txt".
It is required to determine and visualize:

- the number of sentences in the text;
- the number of words in the text
- the number of different words in the text
- the shortest and longest word(s)
- the text without diacritics
- synonyms of the longest word in the text
"""
import nltk
from nltk.corpus import wordnet
from unidecode import unidecode


def noOfSentences():
    """
    Number of phrases in the file
    :return: number of phrases
    """
    with open('data/texts.txt', 'r', encoding='utf-8') as f:
        lines = f.read()
        phrases = lines.replace('!', '.').split('.')
    return len(phrases)


print(noOfSentences())


def noOfWords():
    """
    Number of words in the file
    :return: number of words
    """
    with (open('data/texts.txt', 'r', encoding='utf-8') as f):
        lines = f.read()
        phrase = lines.replace('\n', ' ').replace('!', ' ').replace('.', ' ').replace(':', ' ').replace('”', ' ').replace(',', ' ').split(' ')
        words = [word for word in phrase if word != '']
    return words


print(len(noOfWords()))


def noOfSameWord():
    """
    Number of the different words in the file
    :return: number of different words
    """
    words = noOfWords()
    same_word = set(words)
    return same_word


print(len(noOfSameWord()))


def maximAndMinimLengthWords():
    """
    The longest and the shortest words in the text
    :return: the longest and the shortest words in the text
    """
    words = noOfWords()
    minim = len(min(words, key=len))
    maxim = len(max(words, key=len))
    longest_words = []
    shortest_words = []
    for word in words:
        p = len(word)
        if len(word) == maxim:
            longest_words.append(word)
        if len(word) == minim:
            shortest_words.append(word)

    return set(longest_words), set(shortest_words)


print(maximAndMinimLengthWords())


def removeDiacritics():
    """
    Remove all diacritics from the text
    :return: all text  without any diacritics
    """
    with (open('data/texts.txt', 'r', encoding='utf-8') as f):
        lines = f.read()
        text = unidecode(lines)

    return text


print(removeDiacritics())


def synonymsLongestWord():
    """
    Synonyms of the longest words in the text
    :return: the synonym of the longest words in the text
    """
    words = noOfWords()
    maxim = max(words, key=len)
    synonyms = []

    for synonym in wordnet.synsets(maxim):
        for lemma in synonym.lemma_names():
            synonyms.append(lemma)

    return set(synonyms)


print(synonymsLongestWord())
