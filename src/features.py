from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def extract_bow_features(corpus, max_features=5000):
    """Convert a text corpus into Bag-of-Words count features.

    Args:
        corpus (iterable of str): Preprocessed text documents.
        max_features (int): Maximum vocabulary size to keep.

    Returns:
        tuple: (sparse feature matrix, fitted CountVectorizer instance)
    """
    # CountVectorizer simply counts raw word occurrences per document
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def extract_tfidf_features(corpus, max_features=5000):
    """Convert a text corpus into TF-IDF weighted features.

    Args:
        corpus (iterable of str): Preprocessed text documents.
        max_features (int): Maximum vocabulary size to keep.

    Returns:
        tuple: (sparse feature matrix, fitted TfidfVectorizer instance)
    """
    # sublinear_tf=True applies log scaling to term frequency, which dampens
    # the effect of very frequent words (like "report", "section") that
    # would otherwise dominate purely on raw count
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def top_terms_per_class(X, vectorizer, labels, target_label, top_n=20):
    """Get the top N highest-weighted terms for a specific class.

    Args:
        X: Feature matrix (BoW or TF-IDF).
        vectorizer: Fitted vectorizer used to produce X.
        labels (pd.Series): Class labels aligned with rows of X.
        target_label (str): The class to inspect.
        top_n (int): Number of top terms to return.

    Returns:
        list of tuple: (term, weight) pairs sorted by weight descending.
    """
    # Select only the rows belonging to the target class
    mask = (labels == target_label).values
    # Sum feature weights across all documents in that class
    class_sums = X[mask].sum(axis=0)
    terms = vectorizer.get_feature_names_out()
    # Pair each term with its summed weight, then sort descending
    term_weights = sorted(zip(terms, class_sums.tolist()[0]), key=lambda x: -x[1])
    return term_weights[:top_n]