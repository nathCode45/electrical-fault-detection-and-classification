import pandas as pd
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, callbacks, regularizers, utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

import models as md
from sklearn.metrics import accuracy_score


def plot_history(h, title='model'):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(h.history['loss'], label='train loss')
    plt.plot(h.history.get('val_loss', []), label='val loss')
    plt.legend(); plt.title(title + ' loss')
    plt.subplot(1,2,2)
    plt.plot(h.history.get('accuracy', h.history.get('acc')), label='train acc')
    plt.plot(h.history.get('val_accuracy', []), label='val acc')
    plt.legend(); plt.title(title + ' accuracy')
    plt.show()


def encode_fault(row):
    return (int(row['G']), int(row['C']), int(row['B']), int(row['A']))

def preprocess_standard_scale(X_train, X_test):
    """
    Fit StandardScaler on X_train and transform X_train/X_test.
    Returns: scaler, X_train_scaled, X_test_scaled
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s

def preprocess_poly_and_scale(X_train, X_test, degree=2):
    """
    Polynomial feature expansion followed by standard scaling.
    Returns: pipeline (fitted), X_train_transformed, X_test_transformed
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    # Fit poly on training, transform both
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)
    # Fit scaler on polynomial features
    X_train_ps = scaler.fit_transform(X_train_poly)
    X_test_ps  = scaler.transform(X_test_poly)
    # return a tuple with fitted components in case user wants them
    pipeline = {"poly": poly, "scaler": scaler}
    return X_train_ps, X_test_ps

def evaluate_non_nn_models(X_train, X_test, y_train, y_test, class_names):
    ### LR, DT, RF models

    model_dictionary = { # each value includes tuple: (model_build(), Preprocessing: "Scaling/ScalPoly/None")
        "Logistic Regression (Initial)": (md.build_initial_lr(), "Scaling"),
        "Logistic Regression (Improved)": (md.build_improved_lr(), "ScalPoly"),
        "Decision Tree (Initial)": (md.build_initial_dt(), "Scaling"),
        "Decision Tree (Improved)": (md.build_improved_dt(), "None"),
        "Random Forest (Initial)": (md.build_initial_rf(), "None"),
        "Random Forest (Improved)": (md.build_improved_rf(), "None")
    }



    print(f"{'='*60}")
    print(f"{'SCIKIT-LEARN MODEL EVALUATION':^60}")
    print(f"{'='*60}\n")

    for name, val in model_dictionary.items():
        print(f"Training {name}...")
        
        
        # Apply preprocessing to the model accordingly
        model = val[0]
        X_train_pr = None
        X_test_pr = None
        if val[1]=="Scaling":
            X_train_pr, X_test_pr = preprocess_standard_scale(X_train, X_test)
        elif val[1]=="ScalPoly":
            X_train_pr, X_test_pr = preprocess_poly_and_scale(X_train, X_test)
        else:
            X_train_pr, X_test_pr = X_train, X_test
        


        # Fit the models to the training data
        model.fit(X_train_pr, y_train)
        
        # Generate predictions on the unseen test set
        y_pred = model.predict(X_test_pr)
        
        # C. EVALUATE: Compare predictions (y_pred) to actuals (y_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"--> Accuracy: {acc:.4f}")
        
        # Print the full classification report (Precision, Recall, F1)
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        print("-" * 60)

def evaluate_nn_models(X_train, X_test, y_train, y_test, class_names, improved=True):

    model = None
    if not improved:
        model = md.build_initial_mlp(X_train.shape[1], len(class_names))
    else:
        model = md.build_improved_mlp(X_train.shape[1], len(class_names))

    model.compile(

        loss='sparse_categorical_crossentropy',   # because y are integer labels

        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),

        metrics=['accuracy']

    )

    model.summary()

    early_stop = callbacks.EarlyStopping( # avoids overfitting by halting training if validation loss stops improving for 12 epochs
    monitor='val_loss',
    patience=12,
    restore_best_weights=True)

    check = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=64,
        callbacks=[early_stop, check],
        # class_weight=class_weights,   # uncomment if you computed class_weights
        verbose=2
    )

    

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    plot_history(history, title=("Improved Neural Network Model" if improved else "Initial Neural Network Model"))
    
 


def main():
    electrical_data = pd.read_csv("../../data/classification_dataset.csv")

    print(electrical_data.describe())

    fault_tups = electrical_data.apply(encode_fault, axis=1)
    labels, uniques = pd.factorize(fault_tups)  # labels are integers 0..(n_classes-1)
    class_names = [ "G{}C{}B{}A{}".format(*t) for t in uniques ]
    n_classes = len(uniques)
    print("Unique label tuples (index -> tuple):")
    for i, t in enumerate(uniques):
        print(i, t)

    X = electrical_data[['Ia','Ib','Ic','Va','Vb','Vc']].values.astype(np.float32)
    y = labels  # integer labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )


    evaluate_nn_models(X_train, X_test, y_train, y_test, class_names, improved=True)



    

    

if __name__ == "__main__":
    main()
