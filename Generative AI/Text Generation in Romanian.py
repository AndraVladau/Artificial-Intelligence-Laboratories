"""
Generation in Romanian:
Implement a system that transforms a text (corpus) into a Markov chain and use it to generate
a proverb or a poem in Romanian (use proverbRo.txt or poezieRo.txt files)
Option 1 – Implement a single-state Markov chain or
Option 2 – Implement an n-state Markov chain
*(state = the number of words based on which the prediction is made)
"""

from MarkovChain import MyMarkovChain


def generateProverb():
    def load_corpus(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text_corpus = file.read()
        return text_corpus

    corpus = load_corpus('data/proverbs.txt')

    markov = MyMarkovChain(n=4)
    markov.train(corpus)

    generated_proverb = markov.generate_text(length=20)
    print("Generated Proverb:", generated_proverb)


def generatePoem():
    def load_corpus(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            poem_corpus = file.readlines()
        return poem_corpus

    corpus = load_corpus('data/complete_corpus.txt')

    markov = MyMarkovChain(n=4)
    markov.train(corpus)

    generated_poem = markov.generate_poem(lines=10, words_per_line=10)
    print("\nGenerated Poem:\n", generated_poem)


generateProverb()
generatePoem()
