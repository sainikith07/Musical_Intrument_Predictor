import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from prepare_dataset_multilabel import X_train, X_test, y_train, y_test, CLASSES

print("Training for instruments:", CLASSES)
print("Number of classes:", len(CLASSES))

# =========================
# BUILD OPTIMIZED MODEL
# =========================
model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=(128, 128, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(CLASSES), activation='sigmoid')
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =========================
# EVALUATE
# =========================
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================
model.save("instrument_multilabel.keras")
print("Model saved successfully as instrument_multilabel.keras")
