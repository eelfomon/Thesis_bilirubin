
class X:
    def __init__(self, a, b, c=0.001):
        self.m = tf.keras.models.Sequential()
        self.B(a)
        self.C(b, c)

    def B(self, a):
        self.m.add(Conv2D(16, (3,3), activation='relu', input_shape=(480,480,3)))
        self.m.add(BatchNormalization())
        self.m.add(MaxPooling2D(pool_size=(2,2)))
        for d in a:
            self.m.add(Conv2D(d, (3,3), activation='relu'))
            self.m.add(BatchNormalization())
            self.m.add(MaxPooling2D(pool_size=(2,2)))
            self.m.add(Dropout(0.2))
        self.m.add(Flatten())
        self.m.add(Dense(512, activation='relu'))
        self.m.add(Dropout(0.3))
        self.m.add(Dense(256, activation='relu'))
        self.m.add(Dropout(0.3))
        self.m.add(Dense(128, activation='relu'))
        self.m.add(Dropout(0.3))
        self.m.add(Dense(1, activation='linear'))

    def C(self, b, c):
        if b == 'adam': e = Adam(learning_rate=c)
        elif b == 'sgd': e = SGD(learning_rate=c, momentum=0.9)
        elif b == 'rmsprop': e = RMSprop(learning_rate=c)
        else: raise ValueError("Invalid opt")
        self.m.compile(optimizer=e, loss='mse', metrics=['mae'])

    def D(self, x, y, e=100, v=None, cb=None):
        return self.m.fit(x, y, epochs=e, validation_data=v, callbacks=cb, verbose=0)

    def E(self, x, y):
        p = np.squeeze(self.m.predict(x))
        return mean_squared_error(y, p), mean_absolute_error(y, p), r2_score(y, p)

    def F(self, p):
        self.m.save(p)

    def G(self, p):
        self.m = tf.keras.models.load_model(p)

    def H(self):
        self.m.summary()

    def I(self, h):
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.plot(h.history['loss'], label='Train Loss')
        plt.plot(h.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(h.history['mae'], label='Train MAE')
        plt.plot(h.history['val_mae'], label='Validation MAE')
        plt.title('MAE')
        plt.legend()
        plt.show()

def J(e, c):
    return c * 0.1 if e % 10 == 0 and e != 0 else c

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

def M1(): return X([32,64,128,256,512,1024], 'adam', 0.001)
def M2(): return X([64,128,256,512,1024,2048], 'sgd', 0.01)
def M3(): return X([32,64,128,256,512,1024], 'rmsprop', 0.001)
def M4(): return X([16,32,64,128,256,512], 'adam', 0.0001)
def M5(): return X([64,128,256,512,1024,2048], 'sgd', 0.001)