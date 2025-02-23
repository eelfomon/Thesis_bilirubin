
def train_and_evaluate(model, X_train, X_test, Y_train, Y_test):
    history = model.D(X_train, Y_train, e=50, v=(X_test, Y_test), cb=[early_stopping, tf.keras.callbacks.LearningRateScheduler(J)])
    mse, mae, r2 = model.E(X_test, Y_test)
    print(f"MSE: {mse}, MAE: {mae}, R2 Score: {r2}")
    model.I(history)