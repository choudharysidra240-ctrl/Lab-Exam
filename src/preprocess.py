import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load English stopwords once at module level so we don't reload them on every function call
stop_words = set(stopwords.words('english'))

# PorterStemmer chosen over lemmatization: faster, and legal terms
# (dispute/disputed, comply/compliance) have simple enough morphology
# that stemming gives adequate normalization without needing a full dictionary lookup
stemmer = PorterStemmer()


def preprocess_text(text):
    """Clean and normalize raw notice text for feature extraction.

    Args:
        text (str): Raw input text from the 'notice' column.

    Returns:
        str: Cleaned, lowercased, stemmed text with stopwords removed.
    """
    # Step 1: Remove HTML tags (defensive — dataset may not have any, but exam requires this step)
    text = re.sub(r'<.*?>', '', text)

    # Step 2: Lowercase everything so "Contract" and "contract" are treated as the same token
    text = text.lower()

    # Step 3: Remove punctuation since it carries no class-distinguishing signal for this task
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 4: Tokenize into individual words so we can filter/stem word by word
    tokens = word_tokenize(text)

    # Step 5: Remove stopwords (the, is, and, etc.) — these appear in every class equally
    # and would just add noise to the feature space
    tokens = [t for t in tokens if t not in stop_words]

    # Step 6: Stem each remaining token to collapse word variants to a common root
    tokens = [stemmer.stem(t) for t in tokens]

    # Rejoin tokens into a single string so vectorizers (CountVectorizer/TfidfVectorizer)
    # can process it as normal text
    return ' '.join(tokens)