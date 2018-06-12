# -*- coding: utf-8 -*-
"""
 Evaluation functions
"""
import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from eval_datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW
from eval_datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS, fetch_ESSLI_1a, fetch_ESSLI_2b, \
    fetch_ESSLI_2c
from analogy import *
from six import iteritems
from embedding import Embedding

logger = logging.getLogger(__name__)





def evaluate_analogy(w, X, y, method="add", k=None, category=None, batch_size=100):
    """
    Simple method to score embedding using SimpleAnalogySolver
    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.
    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings"
    X : array-like, shape (n_samples, 3)
      Analogy questions.
    y : array-like, shape (n_samples, )
      Analogy answers.
    k : int, default: None
      If not None will select k top most frequent words from embedding
    batch_size : int, default: 100
      Increase to increase memory consumption and decrease running time
    category : list, default: None
      Category of each example, if passed function returns accuracy per category
      in addition to the overall performance.
      Analogy datasets have "category" field that can be supplied here.
    Returns
    -------
    result: dict
      Results, where each key is for given category and special empty key "" stores
      summarized accuracy across categories
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    assert category is None or len(category) == y.shape[0], "Passed incorrect category list"

    solver = SimpleAnalogySolver(w=w, method=method, batch_size=batch_size, k=k)
    y_pred = solver.predict(X)

    if category is not None:
        results = OrderedDict({"all": np.mean(y_pred == y)})
        count = OrderedDict({"all": len(y_pred)})
        correct = OrderedDict({"all": np.sum(y_pred == y)})
        for cat in set(category):
            results[cat] = np.mean(y_pred[category == cat] == y[category == cat])
            count[cat] = np.sum(category == cat)
            correct[cat] = np.sum(y_pred[category == cat] == y[category == cat])

        return pd.concat([pd.Series(results, name="accuracy"),
                          pd.Series(correct, name="correct"),
                          pd.Series(count, name="count")],
                         axis=1)
    else:
        return np.mean(y_pred == y)



def evaluate_on_semeval_2012_2(w):
    """
    Simple method to score embedding using SimpleAnalogySolver
    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.
    Returns
    -------
    result: pandas.DataFrame
      Results with spearman correlation per broad category with special key "all" for summary
      spearman correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    data = fetch_semeval_2012_2()
    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    categories = data.y.keys()
    results = defaultdict(list)
    for c in categories:
        # Get mean of left and right vector
        prototypes = data.X_prot[c]
        prot_left = np.mean(np.vstack(w.get(word, mean_vector) for word in prototypes[:, 0]), axis=0)
        prot_right = np.mean(np.vstack(w.get(word, mean_vector) for word in prototypes[:, 1]), axis=0)

        questions = data.X[c]
        question_left, question_right = np.vstack(w.get(word, mean_vector) for word in questions[:, 0]), \
                                        np.vstack(w.get(word, mean_vector) for word in questions[:, 1])

        scores = np.dot(prot_left - prot_right, (question_left - question_right).T)

        c_name = data.categories_names[c].split("_")[0]
        # NaN happens when there are only 0s, which might happen for very rare words or
        # very insufficient word vocabulary
        cor = scipy.stats.spearmanr(scores, data.y[c]).correlation
        results[c_name].append(0 if np.isnan(cor) else cor)

    final_results = OrderedDict()
    final_results['all'] = sum(sum(v) for v in results.values()) / len(categories)
    for k in results:
        final_results[k] = sum(results[k]) / len(results[k])
    return pd.Series(final_results)



def evaluate_similarity(w, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs
    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.
    X: array, shape: (n_samples, 2)
      Word pairs
    y: vector, shape: (n_samples,)
      Human ratings
    Returns
    -------
    cor: float
      Spearman correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    missing_words = 0
    miss = []
    words = w.vocabulary.word_id
    for query in X:
        for query_word in query:
            if query_word not in words:
                miss.append(query_word)
                missing_words += 1
    #if missing_words > 0:
    #    print("Missing {} words. Will replace them with mean vector".format(missing_words))
    #    print (miss)


    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A = np.vstack(w.get(word, mean_vector) for word in X[:, 0])
    B = np.vstack(w.get(word, mean_vector) for word in X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation


def evaluate_on_all(w):
    """
    Evaluate Embedding on all fast-running benchmarks
    Parameters
    ----------
    w: Embedding or dict
      Embedding to evaluate.
    Returns
    -------
    results: pandas.DataFrame
      DataFrame with results, one per column.
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    # Calculate results on similarity
    print("Calculating similarity benchmarks")
    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
    }

    similarity_results = {}

    for name, data in iteritems(similarity_tasks):
        similarity_results[name] = evaluate_similarity(w, data.X, data.y)
        print("Spearman correlation of scores on {} {}".format(name, similarity_results[name]))

    # Calculate results on analogy
    print("Calculating analogy benchmarks")
    analogy_tasks = {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }

    analogy_results = {}

    for name, data in iteritems(analogy_tasks):
        analogy_results[name] = evaluate_analogy(w, data.X, data.y)
        print("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))

    analogy_results["SemEval2012_2"] = evaluate_on_semeval_2012_2(w)['all']
    print("Analogy prediction accuracy on {} {}".format("SemEval2012", analogy_results["SemEval2012_2"]))

    # Calculate results on categorization

    analogy = pd.DataFrame([analogy_results])
    sim = pd.DataFrame([similarity_results])
    results = sim.join(analogy)

    return results
