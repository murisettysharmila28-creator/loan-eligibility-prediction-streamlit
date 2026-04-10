from src.config import DATA_PATH
from src.data_loader import load_data
from src.train import train_and_select_best_model
from src.validate import run_cross_validation, compare_split_ratios


def main():
    print("Starting training pipeline...")

    df = load_data(DATA_PATH)

    _, best_model_name, best_score = train_and_select_best_model(df)

    print("\nTraining completed!")
    print(f"Best model: {best_model_name}")
    print(f"Best test accuracy: {best_score:.4f}")

    print("\nRunning 5-fold cross-validation...")
    cv_scores, avg_cv_accuracy, best_cv_accuracy, best_split_index = run_cross_validation(df)

    print("Cross-validation accuracy scores:", cv_scores)
    print(f"Average CV Accuracy: {avg_cv_accuracy:.4f}")
    print(f"Best Fold Accuracy: {best_cv_accuracy:.4f}")
    print(f"Best-Performing Fold: Fold {best_split_index}")

    print("\nComparing train-test split ratios...")
    split_results, best_ratio, best_ratio_score = compare_split_ratios(df)

    print("Split Ratio Results:", split_results)
    print(f"Best Split Ratio: {best_ratio}")
    print(f"Best Split Ratio Accuracy: {best_ratio_score:.4f}")


if __name__ == "__main__":
    main()