from src.config import DATA_PATH
from src.data_loader import load_data
from src.train import train_and_select_best_model


def main():
    print("Starting training pipeline...")

    df = load_data(DATA_PATH)
    _, best_model_name, best_score = train_and_select_best_model(df)

    print("\nTraining completed!")
    print(f"Best model: {best_model_name}")
    print(f"Best test accuracy: {best_score:.4f}")


if __name__ == "__main__":
    main()