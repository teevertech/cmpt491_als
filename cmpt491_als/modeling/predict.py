from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from .sand_datasets import SANDDataset
from cmpt491_als.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def predict_dataset(model, dataloader, device):
    """
    Run inference on entire dataset and collect predictions.

    Returns:
        tuple: (predictions, true_labels, subject_ids, audio_tasks)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_subject_ids = []
    all_audio_tasks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_values=input_values)
            logits = outputs.logits

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_subject_ids.extend(batch['subject_id'])
            all_audio_tasks.extend(batch['audio_task'])

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_subject_ids, all_audio_tasks


def create_confusion_matrix_plot(y_true, y_pred, class_names, save_path):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")


def analyze_per_task_performance(y_true, y_pred, audio_tasks, class_names):
    """Analyze performance per audio task."""
    task_results = {}

    unique_tasks = list(set(audio_tasks))

    for task in unique_tasks:
        task_indices = [i for i, t in enumerate(audio_tasks) if t == task]
        task_true = [y_true[i] for i in task_indices]
        task_pred = [y_pred[i] for i in task_indices]

        if len(task_true) > 0:
            accuracy = accuracy_score(task_true, task_pred)
            task_results[task] = {
                'accuracy': accuracy,
                'n_samples': len(task_true)
            }

    return task_results


def analyze_per_subject_performance(y_true, y_pred, subject_ids):
    """Analyze performance per subject (majority vote)."""
    subject_results = {}
    unique_subjects = list(set(subject_ids))

    subject_true_labels = []
    subject_pred_labels = []

    for subject in unique_subjects:
        subject_indices = [i for i, s in enumerate(subject_ids) if s == subject]
        subject_preds = [y_pred[i] for i in subject_indices]
        subject_trues = [y_true[i] for i in subject_indices]

        # Majority vote for prediction
        pred_majority = max(set(subject_preds), key=subject_preds.count)
        true_label = subject_trues[0]  # Should be same for all samples from subject

        subject_results[subject] = {
            'true_label': true_label,
            'pred_label': pred_majority,
            'n_samples': len(subject_indices),
            'agreement': subject_preds.count(pred_majority) / len(subject_preds)
        }

        subject_true_labels.append(true_label)
        subject_pred_labels.append(pred_majority)

    subject_accuracy = accuracy_score(subject_true_labels, subject_pred_labels)

    return subject_results, subject_accuracy


@app.command()
def evaluate(
    test_data_dir: Path = PROCESSED_DATA_DIR / "SAND" / "task1" / "val",
    metadata_path: Path = INTERIM_DATA_DIR / "sand_dataset.csv",
    model_path: Path = MODELS_DIR / "ast" / "best_model",  # Updated to use model subdirectory
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    results_dir: Path = Path("results"),
    batch_size: int = 16,
    device: str = "auto"
):
    """
    Comprehensive evaluation of trained model.

    Performs detailed analysis including:
    - Overall classification metrics
    - Confusion matrix
    - Per-class performance
    - Per-audio-task performance
    - Per-subject performance (majority vote)
    """

    # Setup device with MPS support
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model and feature extractor
    logger.info(f"Loading model from {model_path}")
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
    model = ASTForAudioClassification.from_pretrained(model_path)
    model = model.to(device)

    # Create dataset and dataloader
    dataset = SANDDataset(
        data_dir=test_data_dir,
        metadata_csv=metadata_path,
        feature_extractor=feature_extractor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda")
    )

    logger.info(f"Evaluating on {len(dataset)} samples...")

    # Run inference
    y_pred, y_true, y_probs, subject_ids, audio_tasks = predict_dataset(model, dataloader, device)

    # Class names (defined locally since they're only used here)
    class_names = ['ALS-1', 'ALS-2', 'ALS-3', 'ALS-4', 'Healthy']

    # Overall metrics
    logger.info("\n" + "="*50)
    logger.info("OVERALL PERFORMANCE")
    logger.info("="*50)

    overall_accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    logger.info(f"\nClassification Report:\n{report}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    logger.info("\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name:8s}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

    # Confusion matrix
    confusion_matrix_path = results_dir / "confusion_matrix.png"
    create_confusion_matrix_plot(y_true, y_pred, class_names, confusion_matrix_path)

    # Per-task analysis
    logger.info("\n" + "="*50)
    logger.info("PER-AUDIO-TASK PERFORMANCE")
    logger.info("="*50)

    task_results = analyze_per_task_performance(y_true, y_pred, audio_tasks, class_names)
    for task, results in task_results.items():
        logger.info(f"{task:12s}: Accuracy={results['accuracy']:.4f} (n={results['n_samples']})")

    # Per-subject analysis (majority vote)
    logger.info("\n" + "="*50)
    logger.info("PER-SUBJECT PERFORMANCE (Majority Vote)")
    logger.info("="*50)

    subject_results, subject_accuracy = analyze_per_subject_performance(y_true, y_pred, subject_ids)
    logger.info(f"Subject-level Accuracy: {subject_accuracy:.4f}")

    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'subject_id': subject_ids,
        'audio_task': audio_tasks,
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_class': [class_names[i] for i in y_true],
        'predicted_class': [class_names[i] for i in y_pred],
        'correct': y_true == y_pred
    })

    # Add probability columns
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = y_probs[:, i]

    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Detailed predictions saved to {predictions_path}")

    # Save summary results
    summary_results = {
        'overall_accuracy': overall_accuracy,
        'subject_accuracy': subject_accuracy,
        'per_class_metrics': {
            class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            } for i in range(len(class_names))
        },
        'per_task_accuracy': task_results
    }

    # Save as JSON
    import json
    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    logger.success("Evaluation complete!")
    logger.info(f"Results saved to {results_dir}")


@app.command()
def predict_test_set(
    test_data_dir: Path = INTERIM_DATA_DIR / "SAND" / "task1" / "testing",
    test_metadata_path: Path = INTERIM_DATA_DIR / "sand_test_dataset.csv",
    model_path: Path = MODELS_DIR / "ast" / "best_model",
    output_path: Path = Path("competition_submission.csv"),
    batch_size: int = 16,
    device: str = "auto"
):
    """
    Generate predictions for competition test set using raw audio.
    """
    # Setup device with MPS support
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Load model and feature extractor
    logger.info(f"Loading model from {model_path}")
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
    model = ASTForAudioClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Load test metadata
    test_df = pd.read_csv(test_metadata_path)
    logger.info(f"Loaded test metadata: {len(test_df)} subjects")

    # Create test dataset using raw audio files
    test_dataset = SANDDataset(
        data_dir=test_data_dir,
        metadata_csv=test_metadata_path,
        feature_extractor=feature_extractor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda")
    )

    logger.info(f"Predicting on {len(test_dataset)} test samples...")

    # Run predictions
    all_preds = []
    all_probs = []
    all_subject_ids = []
    all_audio_tasks = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Prediction"):
            input_values = batch['input_values'].to(device)

            outputs = model(input_values=input_values)
            logits = outputs.logits

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_subject_ids.extend(batch['subject_id'])
            all_audio_tasks.extend(batch['audio_task'])

    # Class names for mapping
    class_names = ['ALS-1', 'ALS-2', 'ALS-3', 'ALS-4', 'Healthy']

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'subject_id': all_subject_ids,
        'audio_task': all_audio_tasks,
        'predicted_label': all_preds,
        'predicted_class': [class_names[i] for i in all_preds],
    })

    # Add probability columns
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = [probs[i] for probs in all_probs]

    # Subject-level predictions (majority vote)
    subject_predictions = []
    unique_subjects = predictions_df['subject_id'].unique()

    for subject in unique_subjects:
        subject_preds = predictions_df[predictions_df['subject_id'] == subject]['predicted_label'].tolist()
        # Majority vote
        majority_pred = max(set(subject_preds), key=subject_preds.count)
        majority_class = class_names[majority_pred]

        # Confidence (proportion of tasks agreeing with majority)
        confidence = subject_preds.count(majority_pred) / len(subject_preds)

        subject_predictions.append({
            'ID': subject,
            'predicted_class_numeric': majority_pred,
            'predicted_class': majority_class,
            'confidence': confidence,
            'n_tasks': len(subject_preds)
        })

    # Save detailed predictions
    predictions_df.to_csv(output_path.with_suffix('.detailed.csv'), index=False)

    # Save competition submission format
    submission_df = pd.DataFrame(subject_predictions)
    submission_df = submission_df[['ID', 'predicted_class_numeric']].rename(columns={
        'predicted_class_numeric': 'Class'
    })
    submission_df.to_csv(output_path, index=False)

    logger.success(f"Test predictions complete!")
    logger.info(f"Detailed predictions: {output_path.with_suffix('.detailed.csv')}")
    logger.info(f"Competition submission: {output_path}")
    logger.info(f"Predicted {len(submission_df)} subjects")

    # Show prediction distribution
    pred_counts = submission_df['Class'].value_counts().sort_index()
    logger.info("Prediction distribution:")
    for class_idx, count in pred_counts.items():
        logger.info(f"  {class_names[class_idx]}: {count} subjects")


if __name__ == "__main__":
    app() # predict.py
