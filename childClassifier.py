import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_data(data_dir, img_size=224):

    X = []
    y = []

    # Load child images
    child_dir = os.path.join(data_dir, 'children')
    for img_file in os.listdir(child_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(child_dir, img_file)
            # Read as grayscale (single channel)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize to required dimensions
                img = cv2.resize(img, (img_size, img_size))
                # Normalize pixel values to [0, 1]
                img = img / 255.0
                X.append(img)
                y.append(0)  # 0 for child

    # Load adult images
    adult_dir = os.path.join(data_dir, 'adults')
    for img_file in os.listdir(adult_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(adult_dir, img_file)
            # Read as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0
                X.append(img)
                y.append(1)  # 1 for adult

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Reshape to have a single channel (grayscale)
    X = X.reshape(X.shape[0], img_size, img_size, 1)

    return X, y


def create_vgg16_ncnn_model(input_shape=(105, 105, 1)):
    inputs = Input(shape=input_shape)

    # Convert grayscale to RGB format for VGG16
    x_rgb = Concatenate(axis=-1)([inputs, inputs, inputs])  # (105, 105, 3)

    # Base VGG16
    base_model = VGG16(include_top=False, input_tensor=x_rgb, input_shape=(105, 105, 3))
    base_model.trainable = False

    x = base_model.output

    # NCNN Module (custom branches)
    branch_a = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    branch_b = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    branch_b = Conv2D(128, (3, 3), padding='same', activation='relu')(branch_b)
    branch_c = AveragePooling2D(pool_size=(1, 1), padding='same')(x)
    branch_c = Conv2D(128, (3, 3), padding='same', activation='relu')(branch_c)

    # Merge all branches
    merged = Concatenate(axis=-1)([branch_a, branch_b, branch_c])
    gap = GlobalAveragePooling2D()(merged)

    # Fully connected classifier
    fc1 = Dense(2048, activation='relu')(gap)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(2048, activation='relu')(fc1)
    fc2 = Dropout(0.4)(fc2)
    fc3 = Dense(512, activation='relu')(fc2)
    output = Dense(2, activation='softmax')(fc3)

    model = Model(inputs=inputs, outputs=output)

    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_training_history(history):

    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()



def predict_face(image_path, model, img_size=105):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  
    img = np.expand_dims(img, axis=-1)  

    prediction = model.predict(img)[0]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    class_name = 'Adult' if predicted_class == 1 else 'Child'

    print(f"Image: {image_path}")
    print(f"Prediction: {class_name} (Confidence: {confidence:.2f})")

    return class_name, confidence


def main():
    # Parameters
    img_size = 105
    batch_size = 64
    epochs = 15
    data_dir = 'yarab'
    num_folds = 3

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(data_dir, img_size)
    print(f"Loaded {len(X)} images: {np.sum(y == 0)} child, {np.sum(y == 1)} adult")
    y_cat = to_categorical(y, num_classes=2)

    # Initialize  K-Fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_no = 1
    for train_index, val_index in skf.split(X, y):
        print(f'\nTraining for Fold {fold_no}...')

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_cat[train_index], y_cat[val_index]

        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        ros = RandomOverSampler(random_state=42)
        X_train_flat_resampled, y_train_resampled = ros.fit_resample(X_train_flat, y_train)

        X_train_resampled = X_train_flat_resampled.reshape((-1, img_size, img_size, 1))
        y_train_resampled_cat = to_categorical(y_train_resampled, num_classes=2)

        print(
            f"Oversampled training set: {X_train_resampled.shape[0]} samples (Child: {np.sum(y_train_resampled == 0)}, Adult: {np.sum(y_train_resampled == 1)})")
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
        )
        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(X_train_resampled, y_train_resampled_cat, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

        
        model = create_vgg16_ncnn_model(input_shape=(img_size, img_size, 1))

        # Callbacks
        checkpoint = ModelCheckpoint(f'best_model_fold_{fold_no}.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)

        # Class weights
        class_weight_dict = {
            0: 1.5,  # Child
            1: 1.0   # Adult
        }

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            class_weight=class_weight_dict
        )

        # Save training plots for each fold
        plot_training_history(history)

        # Evaluate fold
        print(f"Evaluating Fold {fold_no}...")
        y_pred_prob = model.predict(X_val)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)

        print("\nClassification Report:")
        print(classification_report(y_val_classes, y_pred, target_names=['Child', 'Adult']))

        plot_confusion_matrix(y_val_classes, y_pred, classes=['Child', 'Adult'])

        fold_no += 1

    print("\nAll folds completed.")


if __name__ == "__main__":
    main()