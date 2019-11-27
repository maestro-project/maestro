import tensorflow as tf

def my_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, 3, input_shape=[28,28,1]),
        tf.keras.layers.Conv2D(1, 3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train.shape)
    x_train = tf.reshape(x_train, [x_train.shape[0], 28, 28, 1])
    x_test = tf.reshape(x_test, [x_test.shape[0], 28, 28, 1])

    model = my_model()

    print(model.summary())
    model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                                        metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=16,epochs=5)

    print("Now begin evaluation")
    model.evaluate(x_test,  y_test, verbose=2)
    print("Using tf version: " + tf.__version__)
