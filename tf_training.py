
class DataGenerator:
    def __init__(self, dataframe, batch_size=32, target_size=(224, 224), shuffle=True, seed=42):
        self.df = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.seed = seed
        self.datagen = self._create_datagen()

    def _create_datagen(self):
        return ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rescale=1./255 if np.random.rand() > 0.5 else None,
            rotation_range=np.random.choice([30, 40, 50]),
            width_shift_range=np.random.uniform(0.1, 0.3),
            height_shift_range=np.random.uniform(0.1, 0.3),
            shear_range=np.random.uniform(0.1, 0.3),
            zoom_range=np.random.uniform(0.1, 0.3),
            horizontal_flip=bool(np.random.randint(0, 2)),
            fill_mode=np.random.choice(['nearest', 'reflect', 'wrap'])
        )

    def get_generator(self):
        return self.datagen.flow_from_dataframe(
            dataframe=self.df,
            x_col='Filepath',
            y_col='Bilirubin',
            target_size=self.target_size,
            class_mode='raw',
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed
        )

class ModelTrainer:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        base_model = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        for layer in base_model.layers:
            layer.trainable = False if np.random.rand() > 0.5 else True

        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        prediction = Dense(1, activation='linear')(x)

        model = Model(inputs=base_model.input, outputs=prediction)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='mean_squared_error',
                      metrics=['mae'])
        return model

    def train(self, train_generator, val_generator, epochs=50):
        start = datetime.now()
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            steps_per_epoch=5,
            validation_steps=32,
            verbose=2
        )
        duration = datetime.now() - start
        print("Training completed in time:", duration)
        self._plot_history(history)
        return history

    def _plot_history(self, history):
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def hyperparameter_tuning():
    best_model = None
    best_loss = float('inf')
    for lr in [0.0001, 0.0005, 0.001]:
        trainer = ModelTrainer()
        trainer.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                              loss='mean_squared_error',
                              metrics=['mae'])
        print(f'Testing learning rate: {lr}')
        history = trainer.train(train_gen, test_gen, epochs=5)
        final_loss = min(history.history['val_loss'])
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = trainer.model
    return best_model


train_gen = DataGenerator(train_df).get_generator()
test_gen = DataGenerator(test_df).get_generator()

trainer = ModelTrainer()
model = hyperparameter_tuning()
model.summary()