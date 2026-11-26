"""
Model Evaluation and Metrics Module.
Calculates Confusion Matrix, Accuracy, Precision, Recall, and F1-Score from scratch.
Generates publication-quality visualizations using Matplotlib and Seaborn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_confusion_matrix(y_true: list, y_pred: list, pos_label: str = 'spam') -> dict:
    """
    Calculates True Positives, False Positives, True Negatives, False Negatives from scratch.

    Args:
        y_true (list): Actual labels.
        y_pred (list): Predicted labels.
        pos_label (str): Label considered positive ('spam').

    Returns:
        dict: Counts of TP, FP, TN, FN.
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp == pos_label)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp != pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp != pos_label)

    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


def calculate_metrics(y_true: list, y_pred: list, pos_label: str = 'spam') -> dict:
    """
    Calculates Accuracy, Precision, Recall, and F1-Score from scratch.
    """
    cm = compute_confusion_matrix(y_true, y_pred, pos_label)
    tp, fp, tn, fn = cm['TP'], cm['FP'], cm['TN'], cm['FN']

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }


def plot_confusion_matrix(cm_dict: dict, save_path: str = 'results/confusion_matrix.png'):
    """
    Plots and saves the Confusion Matrix Heatmap using Seaborn.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    matrix = np.array([
        [cm_dict['TN'], cm_dict['FP']],
        [cm_dict['FN'], cm_dict['TP']]
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Pred Ham', 'Pred Spam'],
        yticklabels=['Actual Ham', 'Actual Spam'],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title('Naive Bayes Confusion Matrix (SMS Spam)', fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_class_distribution(df: pd.DataFrame, save_path: str = 'results/spam_distribution.png'):
    """
    Plots the dataset class distribution (Spam vs Ham).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    counts = df['label'].value_counts()

    plt.figure(figsize=(6, 5))
    colors = ['#4C72B0', '#DD8452']
    bars = plt.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 50, f'{yval:,} ({yval/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('SMS Spam Collection Class Distribution', fontsize=13, pad=12)
    plt.xlabel('Message Category', fontsize=11)
    plt.ylabel('Number of Messages', fontsize=11)
    plt.ylim(0, max(counts.values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

