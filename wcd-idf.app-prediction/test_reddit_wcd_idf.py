from gensim.models import KeyedVectors
from vec4ir import Retrieval, WordCentroidDistance

documents = ["The quick brown fox jumps over the happy dog",
             "Computer scientists are lazy lazy lazy",
             "Sometimes, all you need to do is completely make an ass of yourself and laugh it off to realise that "
             "life isn’t so bad after all.",
             "We conclude that word centroid similarity is the best competitor to state-of-the-art retrieval models.",
             "He told us a very exciting adventure story.",
             "I think I will buy the red car, or I will lease the blue one."]


def test_reddit_wcd_idf():
    model = KeyedVectors.load_word2vec_format(
        "model/reddit.en.text.vector")  # Replace with directory to your .vector model file
    wcd = WordCentroidDistance(model.wv)
    retrieval = Retrieval(wcd)
    retrieval.fit(documents)

    while True:
        query = input("Please enter the query:\n")
        if query == "exit":
            break
        else:
            result = retrieval.query(query, return_scores=True)
            print(result)


test_reddit_wcd_idf()
