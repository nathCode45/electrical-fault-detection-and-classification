import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers, utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression



# -------------------------s
# Random Forest
# -------------------------
def build_initial_rf():
    """
    Initial Random Forest.
    Works on raw or lightly preprocessed data (trees are scale-invariant).
    """
    return RandomForestClassifier(n_estimators=10, max_depth=10, random_state=22, n_jobs=-1)

def build_improved_rf():
    """
    Improved Random Forest: more trees and mild complexity control.
    """
    return RandomForestClassifier(
        n_estimators=500,       # more trees for stability
        max_depth=None,
        min_samples_leaf=2,
        max_features='sqrt',    # common setting for RF
        random_state=42,
        n_jobs=-1
    )

# -------------------------
# Decision Tree
# -------------------------
def build_initial_dt():
    """
    Initial Decision Tree (no explicit regularization).
    """
    return DecisionTreeClassifier(
        max_depth= 24,
        random_state=42,
        min_samples_split=12,
        min_samples_leaf=6

    )

def build_improved_dt():
    """
    Regularized Decision Tree with depth/min-samples constraints.
    """
    return DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

# -------------------------
# Logistic Regression (with preprocessing pipelines)
# -------------------------
def build_initial_lr():
    """
    Initial Logistic Regression pipeline: StandardScaler + LogisticRegression.
    Returns a scikit-learn pipeline object that accepts raw X.
    """
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', multi_class='multinomial')
    )
    return pipeline

def build_improved_lr(degree=2, C=1.0, penalty='l2'):
    """
    Improved Logistic Regression: PolynomialFeatures -> StandardScaler -> LogisticRegression.
    degree: polynomial degree (2 by default for pairwise interactions)
    C: inverse regularization strength (smaller = stronger reg)
    """
    pipeline = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=3000, solver='lbfgs',
                           penalty=penalty, C=C, multi_class='multinomial')
    )
    return pipeline

# -------------------------
# MLP (TensorFlow / Keras)
# -------------------------
def build_initial_mlp(input_dim, n_classes, l2_reg=1e-4, compile_model=True):
    """
    Initial MLP architecture (matches your Version 1).
    Assumes inputs have been StandardScaled before training.
    Returns an uncompiled Keras model by default, or compiled if compile_model=True.
    """
    reg = regularizers.l2(l2_reg)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(n_classes, activation='softmax')
    ])
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    return model

def build_improved_mlp(input_dim, n_classes, l2_reg=1e-4, compile_model=True):
    """
    Improved MLP with L2 regularization, extra layer, and stronger dropout.
    Assumes StandardScaler-preprocessed inputs.
    """
    reg = regularizers.l2(l2_reg)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu',),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(32, activation='relu', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dense(n_classes, activation='softmax')
    ])
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    return model
