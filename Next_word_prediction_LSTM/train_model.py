from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


APP_DIR = Path(__file__).resolve().parent


@dataclass
class TrainingConfig:
    corpus_path: Path = APP_DIR / "hamlet.txt"
    model_path: Path = APP_DIR / "next_word_lstm.keras"
    tokenizer_path: Path = APP_DIR / "tokenizer.pickle"
    num_words: int = 4000
    context_window: int = 8
    embedding_dim: int = 128
    lstm_units: int = 128
    dropout_rate: float = 0.2
    batch_size: int = 64
    epochs: int = 20
    test_size: float = 0.2
    random_state: int = 42
    learning_rate: float = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_corpus(corpus_path: Path) -> str:
    return normalize_text(corpus_path.read_text(encoding="utf-8"))


def build_tokenizer(text: str, num_words: int) -> Tokenizer:
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    return tokenizer


def make_training_data(
    tokenizer: Tokenizer,
    text: str,
    context_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    token_ids = tokenizer.texts_to_sequences([text])[0]

    sequences = []
    for index in range(context_window, len(token_ids)):
        sequence = token_ids[index - context_window : index + 1]
        # Skip samples whose target word is outside the kept vocabulary.
        if sequence[-1] == 1:
            continue
        sequences.append(sequence)

    sequence_array = np.asarray(sequences, dtype=np.int32)
    features = sequence_array[:, :-1]
    labels = sequence_array[:, -1]
    return features, labels


def build_model(config: TrainingConfig, vocab_size: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(config.context_window,)),
            Embedding(vocab_size, config.embedding_dim),
            LSTM(config.lstm_units, return_sequences=True),
            Dropout(config.dropout_rate),
            LSTM(config.lstm_units),
            Dense(vocab_size, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )
    return model


def train_next_word_model(config: TrainingConfig):
    set_seed(config.random_state)

    text = load_corpus(config.corpus_path)
    tokenizer = build_tokenizer(text, config.num_words)
    features, labels = make_training_data(tokenizer, text, config.context_window)

    vocab_size = min(config.num_words, len(tokenizer.word_index) + 1)
    x_train, x_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = build_model(config, vocab_size)
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_top5_acc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
        EarlyStopping(
            monitor="val_top5_acc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(config.model_path)
    with config.tokenizer_path.open("wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "config": config,
        "history": history,
        "model": model,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "sequence_count": int(features.shape[0]),
        "train_shape": x_train.shape,
        "val_shape": x_val.shape,
    }


def predict_next_words(
    model: Sequential,
    tokenizer: Tokenizer,
    text: str,
    context_window: int,
    top_k: int = 5,
):
    prompt = normalize_text(text)
    token_list = tokenizer.texts_to_sequences([prompt])[0][-context_window:]
    token_list = pad_sequences([token_list], maxlen=context_window, padding="pre")

    probabilities = model.predict(token_list, verbose=0)[0]
    candidates = []
    for word_index in np.argsort(probabilities)[::-1]:
        word = tokenizer.index_word.get(int(word_index))
        if not word or word == "<oov>":
            continue
        candidates.append((word, float(probabilities[word_index])))
        if len(candidates) == top_k:
            break
    return candidates


if __name__ == "__main__":
    config = TrainingConfig()
    artifacts = train_next_word_model(config)

    print(f"Saved model to: {config.model_path}")
    print(f"Saved tokenizer to: {config.tokenizer_path}")
    print(f"Vocabulary size used for training: {artifacts['vocab_size']}")
    print(f"Training sequences: {artifacts['sequence_count']}")
    print(f"Train split shape: {artifacts['train_shape']}")
    print(f"Validation split shape: {artifacts['val_shape']}")

    sample_prompts = [
        "to be or not to",
        "to be or not to be",
        "the king doth",
        "shall i compare thee",
    ]

    for prompt in sample_prompts:
        print(f"\nPrompt: {prompt}")
        for word, score in predict_next_words(
            artifacts["model"],
            artifacts["tokenizer"],
            prompt,
            config.context_window,
        ):
            print(f"  {word}: {score:.4f}")
