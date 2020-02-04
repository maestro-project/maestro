import tensorflow as tf

layer = tf.keras.layers
seq = tf.keras.models.Sequential

def alexnet():
    model = seq([
        layer.Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
        layer.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),

        layer.Conv2D(256, kernel_size=(5,5), strides=(1,1), activation='relu'),
        layer.MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        layer.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu'),

        layer.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu'),

        layer.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        layer.MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        layer.Flatten(),
        layer.Dense(4096, input_shape=(224*224*3,), activation='relu'),
        layer.Dropout(0.4),

        layer.Dense(4096, activation='relu'),
        layer.Dense(4096, activation='relu'),
        layer.Dropout(0.4),

        layer.Dense(1, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
