import nltk
nltk.download()
from gensim import models
import gensim.downloader as api
import A1_helper


def main():
    sentences = nltk.corpus.brown.sents()
    len(sentences)
    sentences_len = sentences[0]
    print(sentences_len)
    type(sentences)
    sentences_list = list(sentences)
    type(sentences_list)

    wv = api.load("word2vec-google-news-300")

    # You can obtain word embeddings by training Word2vec method on your own corpus
    m_cbow = models.Word2Vec(sentences=sentences_list, sg=0, epochs=5, vector_size=100, min_count=5)

    # The vectors can be saved using in a plain text file:
    m_cbow.wv.save_word2vec_format("my_wv.txt")

    # You can obtain word embeddings by training Word2vec method on your own corpus
    m_skipgram = models.Word2Vec(sentences=sentences_list, sg=1, epochs=5, vector_size=100, min_count=5)

    # The vectors can be saved using in a plain text file:
    m_skipgram.wv.save_word2vec_format("my_wv1.txt")

    # They can be loaded from a file using:
    mv_cbow = models.KeyedVectors.load_word2vec_format("my_wv.txt", binary=False)
    mv_skipgram = models.KeyedVectors.load_word2vec_format("my_wv.txt", binary=False)

    #Task 2:
    # Reduce dimensions of the vectors to 2

    x_vals_cbow, y_vals_cbow, labels_cbow = A1_helper.reduce_dimensions(mv_cbow)
    x_vals_skipgram, y_vals_skipgram, labels_skipgram = A1_helper.reduce_dimensions(mv_skipgram)

    # Visualize

    visualize_cbow = A1_helper.plot_with_matplotlib(x_vals_cbow, y_vals_cbow, labels_cbow,
                                                    ["municipal", "shoot", "delicate", "formerly", "Schweitzer",
                                                     "dread", "yielding", "shivering", "swallow", "Band", "dig",
                                                     "examinations", "coupling", "compelling", "homer", "Coach",
                                                     "ghastly", "Quaker", "pollution", "wrecked"])
    visualize_skipgram = A1_helper.plot_with_matplotlib(x_vals_skipgram, y_vals_skipgram, labels_skipgram,
                                                        ["municipal", "shoot", "delicate", "formerly", "Schweitzer",
                                                         "dread", "yielding", "shivering", "swallow", "Band", "dig",
                                                         "examinations", "coupling", "compelling", "homer", "Coach",
                                                         "ghastly", "Quaker", "pollution", "wrecked"])

    # Task 3:
    pearson_google = wv.evaluate_word_pairs("sim.txt")
    print("", pearson_google)
    pearson_cbow = mv_cbow.evaluate_word_pairs("sim.txt")
    print("", pearson_cbow)
    pearson_skipgram = mv_skipgram.evaluate_word_pairs("sim.txt")
    print("", pearson_skipgram)

    # Task 4:
    #google embedding
    wv.most_similar("municipal")
    wv.most_similar("shoot")
    wv.most_similar("delicate")
    wv.most_similar("dig")
    wv.most_similar("swallow")

    # cbow
    mv_cbow.most_similar("municipal")
    mv_cbow.most_similar("shoot")
    mv_cbow.most_similar("delicate")
    mv_cbow.most_similar("dig")
    mv_cbow.most_similar("swallow")

    # skipgram
    mv_skipgram.most_similar("municipal")
    mv_skipgram.most_similar("shoot")
    mv_skipgram.most_similar("delicate")
    mv_skipgram.most_similar("dig")
    mv_skipgram.most_similar("swallow")


if __name__ == "__main__":
    main()