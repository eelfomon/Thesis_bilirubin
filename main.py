
def main():
    X_train, X_test, Y_train, Y_test = load_data()
    models = [M1(), M2(), M3(), M4(), M5()]

    for i, model in enumerate(models, 1):
        print(f"Training Model {i}...")
        train_and_evaluate(model, X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()