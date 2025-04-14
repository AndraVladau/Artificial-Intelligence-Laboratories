"""
English generation:
a. Use the markovify library (or your implementation from problem 1) to generate a stanza of English poetry
using one of the following corpora (or any other source you find):
    - https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus
    - https://github.com/tnhaider/english-gutenberg-poetry
    - https://www.shakespeares-sonnets.com/all.php

b. Calculate the emotion of the generated text, you can use one of the following resources:
    - Natural Language Toolkit (nltk) SentimentIntensityAnalyzer
    - TextBlob feeling

c. To address the limitations of creativity in the generated poem randomly replace words with synonyms.
Synonyms are required to be obtained using embeddings.
ex. the chosen word is transformed into its embedded form and the closest embedding is chosen which is converted to a string

d. Save the poem that you think is the most successful and send it to a friend.

e. Calculate the BLEU (Bilingual Evaluation Understudy Score) metric for the selected poem
"""

from MarkovChain import MyMarkovChain
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


def main():
    def load_corpus(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            poem_corpus = file.readlines()
        return poem_corpus

    corpus = load_corpus('data/shakespeare-sonnets.txt')

    markov = MyMarkovChain(n=4)
    markov.train(corpus)

    generated_poem = markov.generate_poem(lines=10, words_per_line=10)
    print("Generated Poem:\n", generated_poem)

    synonyms_generated = markov.generate_poem_with_synonyms_from_existing(generated_poem)
    print("\nSynonyms generated:\n", synonyms_generated)

    sia = SentimentIntensityAnalyzer().polarity_scores(generated_poem)
    tb = TextBlob(generated_poem)

    print("\nSentiment Analysis with NLTK:")
    print(f"The Negativity Score: {sia['neg']}, The Neutral Score: {sia['neu']}, The Positivity Score: {sia['pos']}, The Overall Compound Score: {sia['compound']}")

    print("\nSentiment Analysis with TextBlob:")
    print(f"Polarity: {tb.sentiment.polarity}, Subjectivity: {tb.sentiment.subjectivity}")

    bleu_score = markov.calculate_bleu(synonyms_generated, generated_poem)
    print("\nBLEU Score:", bleu_score)


if __name__ == "__main__":
    main()






