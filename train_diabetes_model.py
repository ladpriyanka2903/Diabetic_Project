import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import os

# ================= PATHS =================
train_dir = r"D:/Diabetic Project/Dataset/training"
val_dir   = r"D:/Diabetic Project/Dataset/testing"
checkpoint_path = r"D:/Diabetic Project/model/Vgg16-diabetes-best.h5"

# ================= PARAMETERS =================
img_height, img_width = 224, 224
batch_size = 16
epochs = 30          # ✅ Phase 1 epochs = 30
num_classes = 5      # change if your classes are different

# ================= LOAD DATASETS =================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

# ================= PERFORMANCE OPT =================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ================= DATA AUGMENTATION =================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
])

# ================= LOAD VGG16 BASE =================
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

base_model.trainable = False  # freeze first

# ================= BUILD MODEL =================
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.vgg16.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# ================= COMPILE =================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= CALLBACKS =================
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

# ================= TRAIN PHASE 1 =================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,   # ✅ 30 epochs
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# ================= FINE-TUNING =================
print("🔓 Fine-tuning top VGG16 layers...")

for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,   # ✅ Fine-tuning = 10 epochs
    callbacks=[checkpoint, reduce_lr, early_stop]
)

print("✅ Training completed and best model saved!")

# ================= PLOT =================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc (Phase 1)')
plt.plot(history.history['val_accuracy'], label='Val Acc (Phase 1)')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss (Phase 1)')
plt.plot(history.history['val_loss'], label='Val Loss (Phase 1)')
plt.title('Loss')
plt.legend()

plt.show()
