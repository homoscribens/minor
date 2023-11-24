import argparse
import pathlib
from logging import INFO, FileHandler, StreamHandler, basicConfig, getLogger

import numpy as np
import pandas as pd
import vibrato
import zstandard as zstd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def initialize_logger():
    basicConfig(
        level=INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[FileHandler('logfile.log'), StreamHandler()],
    )
    return getLogger(__name__)


# Usage
logger = initialize_logger()


def get_tfidf_matrix(tokenizer, sys_utter, user_utter, decomp_dim=2):
    tfidf = TfidfVectorizer(
        analyzer=lambda x: [token.surface() for token in tokenizer.tokenize(x)]
    )
    statistic_embeddings_sys = tfidf.fit_transform(sys_utter)
    statistic_embeddings_user = tfidf.fit_transform(user_utter)

    tsvd = TruncatedSVD(n_components=decomp_dim, random_state=42)
    assert (
        decomp_dim <= statistic_embeddings_sys.shape[1]
        or decomp_dim <= statistic_embeddings_user.shape[1]
    )
    logger.info("Fitting TruncatedSVD...")
    statistic_embeddings_2d_sys = tsvd.fit_transform(statistic_embeddings_sys)
    statistic_embeddings_2d_user = tsvd.fit_transform(statistic_embeddings_user)

    utter_embeddings = np.concatenate(
        [statistic_embeddings_2d_sys, statistic_embeddings_2d_user], axis=1
    )

    return utter_embeddings


def main(file_paths, test_size, decomp_dim):
    dctx = zstd.ZstdDecompressor()

    with open('./ipadic-mecab-2_7_0/system.dic.zst', 'rb') as fp:
        with dctx.stream_reader(fp) as dict_reader:
            tokenizer = vibrato.Vibrato(dict_reader.read())

    sys_utters = []
    user_utters = []
    labels = []
    for file_path in tqdm(file_paths):
        df = pd.read_csv(file_path)
        sys_utter = df["system_utterance"].tolist()
        user_utter = df["user_utterance"].fillna("0").tolist()
        label = df["SS_ternary"].tolist()

        sys_utters.extend(sys_utter)
        user_utters.extend(user_utter)
        labels.extend(label)

    utter_embeddings = get_tfidf_matrix(
        tokenizer, sys_utters, user_utters, decomp_dim=decomp_dim
    )
    labels = np.array(labels)
    logger.info(f"utter_embeddings.shape: {utter_embeddings.shape}")
    logger.info(f"labels.shape: {labels.shape}")

    # Assuming data is your feature matrix and labels is your target vector
    X_train, X_test, y_train, y_test = train_test_split(
        utter_embeddings, labels, test_size=test_size, random_state=42
    )

    clf = LogisticRegression(random_state=42).fit(X_train, y_train)
    preds = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    logger.info(f"Accuracy: {score}")
    logger.info(f"Predicted value counts: {np.unique(preds, return_counts=True)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_size', type=float, default=0.1, help='Test size for train-test split'
    )
    parser.add_argument(
        '--decomp_dim',
        type=int,
        default=2,
        help='Number of dimensions for TruncatedSVD',
    )
    args = parser.parse_args()

    folder_path = pathlib.Path('preprocessed')
    file_paths = list(folder_path.glob('*.csv'))
    logger.info(f"file_paths: {file_paths}")

    main(file_paths, args.test_size, args.decomp_dim)
