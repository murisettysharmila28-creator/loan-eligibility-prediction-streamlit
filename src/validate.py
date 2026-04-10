from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

from src.train import preprocess_data
from src.config import TARGET_COLUMN
from src.logger import setup_logger

logger = setup_logger()


def run_cross_validation(df):
    df = preprocess_data(df)

    x = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline,
        x,
        y,
        cv=cv,
        scoring="accuracy"
    )

    avg_cv_accuracy = np.mean(cv_scores)
    best_cv_accuracy = np.max(cv_scores)
    best_split_index = np.argmax(cv_scores) + 1   # +1 because fold numbering starts from 1

    logger.info(f"Cross-validation accuracy scores: {cv_scores}")
    logger.info(f"Average cross-validation accuracy: {avg_cv_accuracy:.4f}")
    logger.info(f"Best fold accuracy: {best_cv_accuracy:.4f}")
    logger.info(f"Best-performing split: Fold {best_split_index}")

    return cv_scores, avg_cv_accuracy, best_cv_accuracy, best_split_index

def compare_split_ratios(df):
    df = preprocess_data(df)

    x = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    split_options = {
        "80:20": 0.2,
        "75:25": 0.25,
        "70:30": 0.3,
    }

    results = {}

    for ratio_label, test_size in split_options.items():
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            stratify=y,
            random_state=42
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000))
        ])

        pipeline.fit(x_train, y_train)
        accuracy = pipeline.score(x_test, y_test)

        results[ratio_label] = accuracy

    best_ratio = max(results, key=results.get)
    best_ratio_score = results[best_ratio]

    logger.info(f"Split ratio comparison: {results}")
    logger.info(f"Best split ratio: {best_ratio} with accuracy {best_ratio_score:.4f}")

    return results, best_ratio, best_ratio_score