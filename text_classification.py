import argparse
import json
import os
import pathlib
from logging import INFO, FileHandler, StreamHandler, basicConfig, getLogger

import numpy as np
import openai
import pandas as pd
import pytorch_lightning as pl
import umap
import vibrato
import zstandard as zstd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from net import NeuralNet, NNDataset


def initialize_logger():
    basicConfig(
        level=INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[FileHandler("logfile.log"), StreamHandler()],
    )
    return getLogger(__name__)


# Usage
logger = initialize_logger()

# OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)


def get_tfidf_matrix(
    tokenizer, sys_utter, user_utter, decomp_dim=100, seed=42
):
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
            [
                statistic_embeddings_sys.toarray(),
                statistic_embeddings_user.toarray(),
            ],
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

    response = client.embeddings.create(
        model="text-embedding-ada-002", input=text
    )
    embeddings = [data.embedding for data in response.data]
    return np.array(embeddings)


def get_ada_embeddings(sys_utter, user_utter, decomp_dim=100, seed=42):
    sys_embeddings = get_embedding(sys_utter)
    user_embeddings = get_embedding(user_utter)

    reducer = umap.UMAP(n_components=decomp_dim, random_state=seed)
    assert (
        decomp_dim <= sys_embeddings.shape[1]
        or decomp_dim <= user_embeddings.shape[1]
    )

    logger.info("Fitting UMAP...")
    sys_embeddings = reducer.fit_transform(sys_embeddings)
    user_embeddings = reducer.fit_transform(user_embeddings)

    utter_embeddings = np.concatenate(
        [sys_embeddings, user_embeddings], axis=1
    )
    return utter_embeddings


def preprocess(df, tokenizer, args):
    sys_utters = df["system_utterance"].tolist()
    user_utters = df["user_utterance"].fillna("-").tolist()
    labels = df[args.label]

    if args.encoder == "tfidf":
        utter_embeddings, tfidf_feature_names = get_tfidf_matrix(
            tokenizer,
            sys_utters,
            user_utters,
            decomp_dim=args.decomp_dim,
            seed=args.seed,
        )
    elif args.encoder == "ada":
        utter_embeddings = get_ada_embeddings(
            sys_utters, user_utters, decomp_dim=args.decomp_dim, seed=args.seed
        )
    elif args.encoder == "None":
        utter_embeddings = None
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    columns_to_drop = pd.Index([])  # Initialize an empty pandas Index

    # Add to the Index the columns that match each pattern
    column_keywords_to_drop = [
        "pcm",
        "F0",
        "voice",
        "[ms]",
        "system",
        "user",
        "SS",
        "TS",
        "TC",
    ]
    for pattern in column_keywords_to_drop:
        columns_to_drop = columns_to_drop.union(
            df.filter(like=pattern).columns
        )

    df = df.drop(columns=columns_to_drop)

    if utter_embeddings is not None and tfidf_feature_names is not None:
        assert len(df) == len(utter_embeddings)
        logger.info(f"utter_embeddings.shape: {utter_embeddings.shape}")
        logger.info(f"labels.shape: {labels.shape}")
        df = pd.concat([df, pd.DataFrame(utter_embeddings)], axis=1)
        df.columns = df.columns.astype(str)
        return df, labels, tfidf_feature_names

    return df, labels, None


def create_dataloader(X, y, batch_size=32):
    dataset = NNDataset(X.to_numpy(), y.to_numpy())
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def train_nn(
    input_size,
    hidden_size,
    num_classes,
    train_loader,
    valid_loader,
    test_loader,
    args,
):
    # Create an instance of the model
    model = NeuralNet(input_size, hidden_size, num_classes, args.lr)

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints_{args.label}",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stopping, checkpoint_callback],
        deterministic=True,
    )

    # Train the model
    trainer.fit(model, train_loader, valid_loader)

    score = trainer.test(model, test_loader)

    return score


def main(args):
    dctx = zstd.ZstdDecompressor()

    with open("./ipadic-mecab-2_7_0/system.dic.zst", "rb") as fp:
        with dctx.stream_reader(fp) as dict_reader:
            tokenizer = vibrato.Vibrato(dict_reader.read())

    dataframes = []
    for file_path in tqdm(args.file_paths):
        df = pd.read_csv(file_path)
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    X_df, labels, feature_names = preprocess(df, tokenizer, args)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
    )

    logger.info(f"X_train.shape: {X_train.shape}")
    logger.info(f"X_test.shape: {X_test.shape}")
    logger.info(f"y_train.shape: {y_train.shape}")
    logger.info(f"y_test.shape: {y_test.shape}")

    # TODO: Add stratify option
    if args.model == "logistic":
        clf = LogisticRegression(
            random_state=args.seed, max_iter=10**17
        ).fit(X_train, y_train)
        # preds = clf.predict(X_test)
        score = clf.score(X_test, y_test)

    elif args.model == "nn":
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=args.test_size,
            random_state=args.seed,
        )
        train_loader = create_dataloader(X_train, y_train, args.batch_size)
        valid_loader = create_dataloader(X_valid, y_valid, args.batch_size)
        test_loader = create_dataloader(X_test, y_test, args.batch_size)

        for batch in train_loader:
            x, y = batch
            print(x.shape)
            print(y.shape)
            break

        input_size = X_train.shape[1]
        hidden_size = 100
        num_classes = len(np.unique(y_train))
        score = train_nn(
            input_size,
            hidden_size,
            num_classes,
            train_loader,
            valid_loader,
            test_loader,
            args,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    logger.info(f"Accuracy: {score}")
    """logger.info(
        f"Predicted value counts: {np.unique(preds, return_counts=True)}"
    )"""

    if args.model == "logistic" and not args.decomp_dim:
        coef = clf.coef_[0]
        weighted_indices = [
            i for i, c in enumerate(coef) if c > args.threshold
        ]
        weighted_features = [feature_names[i] for i in weighted_indices]
        weighted_dict = {feature_names[i]: coef[i] for i in weighted_indices}
        logger.info(f"Selected features: {weighted_features}")
        logger.info(f"Weighted dictionary: {weighted_dict}")

        with open(args.output_path, "w", encoding="utf-8") as fp:
            json.dump(weighted_dict, fp, indent=4)
        logger.info(f"Saved weighted dictionary to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data preprocessing hyperparameters
    parser.add_argument(
        "--label",
        type=str,
        default="SS_ternary",
        help="Label column name in the csv file",
        choices=[
            "SS_ternary",
            "TS_ternary",
            "TC_ternary",
        ],
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Test size for train-test split",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        help="Model for text classification",
        choices=["logistic", "nn"],
    )

    # NN model hyperparameters
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for nn"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for nn"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=30, help="Max epochs for nn"
    )
    parser.add_argument(
        "--patience", type=int, default=2, help="Patience for nn"
    )

    # Logistic regression hyperparameters
    parser.add_argument(
        "--encoder",
        type=str,
        default="tfidf",
        help="Encoder for utterance embeddings",
        choices=["tfidf", "ada", "None"],
    )
    parser.add_argument(
        "--decomp_dim",
        type=int,
        default=100,
        help="Number of dimensions for TruncatedSVD",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Threshold for feature selection",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train-test split"
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    folder_path = pathlib.Path("preprocessed")
    file_paths = list(folder_path.glob("*.csv"))
    logger.info(f"file_paths: {file_paths}")
    args.file_paths = file_paths

    output_folder_path = pathlib.Path("output/texts")
    if not output_folder_path.exists():
        output_folder_path.mkdir(parents=True)
    args.output_path = output_folder_path / f"{args.label}.json"

    pl.seed_everything(args.seed)
    main(args)
