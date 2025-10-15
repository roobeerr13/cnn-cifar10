def train(model, x_train, y_train_cat, epochs=10, validation_split=0.1):
    history = model.fit(
        x_train, y_train_cat,
        epochs=epochs,
        validation_split=validation_split
    )
    return history