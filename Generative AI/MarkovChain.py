import numpy as np
import random
import nltk


class MyMarkovChain:
    def __init__(self, n):
        self.n = n
        self.markov_chain = {}
        self.embeddings, self.word_list = self.load_glove_embeddings('data/embedding-words.txt')

    def load_glove_embeddings(self, glove_file):
        embeddings = {}
        word_list = []
        with open(glove_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                embeddings[word] = vector
                word_list.append(word)
        return embeddings, word_list

    def find_synonym(self, word, threshold=0.5, max_attempts=100):
        if word not in self.embeddings:
            return word
        attempts = 0
        while attempts < max_attempts:
            synonym = random.choice(self.word_list)
            if synonym == word:
                continue
            try:
                similarity = np.dot(self.embeddings[word], self.embeddings[synonym]) / (np.linalg.norm(self.embeddings[word]) * np.linalg.norm(self.embeddings[synonym]))
            except KeyError:
                continue
            if similarity > threshold:
                return synonym
            attempts += 1
        return word

    def generate_text_with_synonyms(self, length, start_state=None):
        if not start_state:
            start_state = random.choice(list(self.markov_chain.keys()))
        output = list(start_state)
        current_state = start_state

        for _ in range(length):
            if current_state not in self.markov_chain:
                break
            next_word_choices = list(self.markov_chain[current_state].keys())
            next_word_weights = list(self.markov_chain[current_state].values())
            next_word = random.choices(next_word_choices, weights=next_word_weights, k=1)[0]
            synonym = self.find_synonym(next_word)
            output.append(synonym)
            current_state = tuple(output[-self.n:])

        return ' '.join(output)

    def train(self, corpus):
        if isinstance(corpus, str):
            corpus = [corpus]
        for text in corpus:
            words = nltk.word_tokenize(text)
            for i in range(len(words) - self.n):
                state = tuple(words[i:i + self.n])
                next_word = words[i + self.n]
                if state not in self.markov_chain:
                    self.markov_chain[state] = {}
                if next_word not in self.markov_chain[state]:
                    self.markov_chain[state][next_word] = 0
                self.markov_chain[state][next_word] += 1

    def generate_poem_with_synonyms_from_existing(self, poem):
        lines = poem.split('\n')
        new_poem = []
        for line in lines:
            new_line = self.generate_line_with_synonyms_from_existing(line)
            new_poem.append(new_line)
        return '\n'.join(new_poem)

    def generate_line_with_synonyms_from_existing(self, line):
        tokens = nltk.word_tokenize(line)
        new_line = []
        for token in tokens:
            if random.random() < 0.5:
                synonym = self.find_synonym(token)
                new_line.append(synonym)
            else:
                new_line.append(token)
        return ' '.join(new_line)

    def generate_text(self, length, start_state=None):
        if not start_state:
            start_state = random.choice(list(self.markov_chain.keys()))
        output = list(start_state)
        current_state = start_state

        for _ in range(length):
            if current_state not in self.markov_chain:
                break
            next_word_choices = list(self.markov_chain[current_state].keys())
            next_word_weights = list(self.markov_chain[current_state].values())
            next_word = random.choices(next_word_choices, weights=next_word_weights, k=1)[0]
            output.append(next_word)
            current_state = tuple(output[-self.n:])

        return ' '.join(output)

    def generate_poem(self, lines, words_per_line):
        poem = []
        for _ in range(lines):
            line = self.generate_line(words_per_line)
            poem.append(line)
        return '\n'.join(poem)

    def generate_line(self, words):
        start_state = random.choice(list(self.markov_chain.keys()))
        output = list(start_state)
        current_state = start_state

        while len(output) < words:
            if current_state not in self.markov_chain:
                break
            next_word_choices = list(self.markov_chain[current_state].keys())
            next_word_weights = list(self.markov_chain[current_state].values())
            next_word = random.choices(next_word_choices, weights=next_word_weights, k=1)[0]
            output.append(next_word)
            current_state = tuple(output[-self.n:])

        return ' '.join(output)

    def calculate_bleu(self, reference_poem, generated_poem):
        reference_tokenized = []
        generated_tokenized = []

        reference_lines = reference_poem.split('\n')
        for line in reference_lines:
            tokens = nltk.word_tokenize(line)
            synonym_tokens = [self.find_synonym(token) for token in tokens]
            reference_tokenized.append(synonym_tokens)

        generated_lines = generated_poem.split('\n')
        for line in generated_lines:
            tokens = nltk.word_tokenize(line)
            synonym_tokens = [self.find_synonym(token) for token in tokens]
            generated_tokenized.append(synonym_tokens)

        precisions = []
        for n in range(1, 5):
            matches = 0
            candidates = 0
            references = 0
            for reference, candidate in zip(reference_tokenized, generated_tokenized):
                reference_ngrams = list(nltk.ngrams(reference, n))
                candidate_ngrams = list(nltk.ngrams(candidate, n))
                matches += sum(1 for ngram in candidate_ngrams if ngram in reference_ngrams)
                candidates += len(candidate_ngrams)
                references += len(reference_ngrams)
            precision = matches / candidates if candidates > 0 else 0
            precisions.append(precision)

        brevity_penalty = min(1.0, len(generated_tokenized) / len(reference_tokenized))
        bleu_score = brevity_penalty * np.exp(np.mean(np.log(precisions)))

        return bleu_score




