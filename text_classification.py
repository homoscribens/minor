import argparse
import pathlib
from logging import INFO, FileHandler, StreamHandler, basicConfig, getLogger

import numpy as np
import openai
import pandas as pd
import umap
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

# OpenAI API key
client = openai.OpenAI(api_key="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")


def get_tfidf_matrix(tokenizer, sys_utter, user_utter, decomp_dim=100, seed=42):
    logger.info("Fitting TfidfVectorizer...")
    tfidf = TfidfVectorizer(
        analyzer=lambda x: [token.surface() for token in tokenizer.tokenize(x)]
    )

    statistic_embeddings_sys = tfidf.fit_transform(sys_utter)
    feature_names_sys = tfidf.get_feature_names_out()
    assert len(feature_names_sys) == statistic_embeddings_sys.shape[1]

    statistic_embeddings_user = tfidf.fit_transform(user_utter)
    feature_names_user = tfidf.get_feature_names_out()
    assert len(feature_names_user) == statistic_embeddings_user.shape[1]

    if not decomp_dim:
        utter_embeddings = np.concatenate(
            [statistic_embeddings_sys.toarray(), statistic_embeddings_user.toarray()],
            axis=1,
        )
        feature_names = np.concatenate([feature_names_sys, feature_names_user])
        return utter_embeddings, feature_names

    tsvd = TruncatedSVD(n_components=decomp_dim, random_state=seed)
    assert (
        decomp_dim <= statistic_embeddings_sys.shape[1]
        or decomp_dim <= statistic_embeddings_user.shape[1]
    )
    logger.info("Fitting TruncatedSVD...")
    statistic_embeddings_sys = tsvd.fit_transform(statistic_embeddings_sys)
    statistic_embeddings_user = tsvd.fit_transform(statistic_embeddings_user)

    utter_embeddings = np.concatenate(
        [statistic_embeddings_sys, statistic_embeddings_user], axis=1
    )
    return utter_embeddings, None


def get_embedding(text):
    logger.info("Getting embeddings...")
    if len(text) > 2048:
        logger.warning(
            "Text length is longer than 2048. Splitting text into two parts."
        )
        text_former = text[:2048]
        text_latter = text[2048:]
        response_former = client.embeddings.create(
            model="text-embedding-ada-002", input=text_former
        )
        response_latter = client.embeddings.create(
            model="text-embedding-ada-002", input=text_latter
        )
        embeddings_former = [data.embedding for data in response_former.data]
        embeddings_latter = [data.embedding for data in response_latter.data]
        embeddings = embeddings_former + embeddings_latter
        return np.array(embeddings)

    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    embeddings = [data.embedding for data in response.data]
    return np.array(embeddings)


def get_ada_embeddings(sys_utter, user_utter, decomp_dim=100, seed=42):
    sys_embeddings = get_embedding(sys_utter)
    user_embeddings = get_embedding(user_utter)

    reducer = umap.UMAP(n_components=decomp_dim, random_state=seed)
    assert (
        decomp_dim <= sys_embeddings.shape[1] or decomp_dim <= user_embeddings.shape[1]
    )

    logger.info("Fitting UMAP...")
    sys_embeddings = reducer.fit_transform(sys_embeddings)
    user_embeddings = reducer.fit_transform(user_embeddings)

    utter_embeddings = np.concatenate([sys_embeddings, user_embeddings], axis=1)
    return utter_embeddings


def main(file_paths, args):
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
        user_utter = df["user_utterance"].fillna("-").tolist()
        label = df[args.label].tolist()

        sys_utters.extend(sys_utter)
        user_utters.extend(user_utter)
        labels.extend(label)

    if args.encoder == 'tfidf':
        utter_embeddings, feature_names = get_tfidf_matrix(
            tokenizer,
            sys_utters,
            user_utters,
            decomp_dim=args.decomp_dim,
            seed=args.seed,
        )
    elif args.encoder == 'ada':
        utter_embeddings = get_ada_embeddings(
            sys_utters, user_utters, decomp_dim=args.decomp_dim, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    labels = np.array(labels)
    logger.info(f"utter_embeddings.shape: {utter_embeddings.shape}")
    logger.info(f"labels.shape: {labels.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        utter_embeddings, labels, test_size=args.test_size, random_state=42
    )

    clf = LogisticRegression(random_state=args.seed, max_iter=10000).fit(
        X_train, y_train
    )
    preds = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    logger.info(f"Accuracy: {score}")
    logger.info(f"Predicted value counts: {np.unique(preds, return_counts=True)}")

    if not args.decomp_dim:
        coef = clf.coef_[0]
        weighted_indices = [i for i, c in enumerate(coef) if c > args.threshold]
        weighted_features = [feature_names[i] for i in weighted_indices]
        logger.info(f"Selected features: {weighted_features}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--label',
        type=str,
        default='SS_ternary',
        help='Label column name in the csv file',
        choices=[
            'SS_ternary',
            'TS_ternary',
            'TC_ternary',
        ],
    )
    parser.add_argument(
        '--test_size', type=float, default=0.1, help='Test size for train-test split'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='tfidf',
        help='Encoder for utterance embeddings',
        choices=[
            'tfidf',
            'ada',
        ],
    )
    parser.add_argument(
        '--decomp_dim',
        type=int,
        default=100,
        help='Number of dimensions for TruncatedSVD',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Threshold for feature selection',
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for train-test split'
    )
    args = parser.parse_args()

    folder_path = pathlib.Path('preprocessed')
    file_paths = list(folder_path.glob('*.csv'))
    logger.info(f"file_paths: {file_paths}")

    main(file_paths, args)
