import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

def evaluate_combined_classification(gt_file, pred_file):
    """
    Evaluate both Action and Function classification together.
    Removes samples if either prediction is N/A or if ground truth is unlabeled.
    """
    
    print("Loading CSV files...")
    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)
    
    print(f"Ground truth samples: {len(gt_df)}")
    print(f"Prediction samples: {len(pred_df)}")
    
    # Merge dataframes
    print("\nMatching samples between files...")
    merged_df = pd.merge(
        gt_df[['sample_name', 'ACTION1_ID', 'ACTION2_ID', 'ACTION3_ID', 'FUNCTION1_ID', 'FUNCTION2_ID']], 
        pred_df[['Video_Name', 'Action_Class', 'Function_Class']], 
        left_on='sample_name', 
        right_on='Video_Name', 
        how='inner'
    )
    
    print(f"Matched samples: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("ERROR: No matching samples found!")
        return
    
    # Convert all IDs to integers
    print("\nProcessing IDs...")
    action_gt_cols = ['ACTION1_ID', 'ACTION2_ID', 'ACTION3_ID']
    function_gt_cols = ['FUNCTION1_ID', 'FUNCTION2_ID']
    
    for col in action_gt_cols + function_gt_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(-1).astype(int)
    
    for col in ['Action_Class', 'Function_Class']:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(-1).astype(int)
    
    # Filtering
    # 1. Unlabeled action GT (all action IDs are -1)
    unlabeled_action_gt = (
        (merged_df['ACTION1_ID'] == -1) & 
        (merged_df['ACTION2_ID'] == -1) & 
        (merged_df['ACTION3_ID'] == -1)
    )
    
    # 2. Unlabeled function GT (both function IDs are -1)
    unlabeled_function_gt = (
        (merged_df['FUNCTION1_ID'] == -1) & 
        (merged_df['FUNCTION2_ID'] == -1)
    )
    
    # 3. Invalid action prediction
    invalid_action_pred = merged_df['Action_Class'] == -1
    
    # 4. Invalid function prediction
    invalid_function_pred = merged_df['Function_Class'] == -1
    
    # Combined: remove if ANY of these conditions is true
    remove_mask = unlabeled_action_gt | unlabeled_function_gt | invalid_action_pred | invalid_function_pred
    
    print(f"\n{'='*70}")
    print("FILTERING SAMPLES:")
    print(f"{'='*70}")
    print(f"Total matched samples: {len(merged_df)}")
    print(f"Unlabeled Action GT (all action IDs = -1): {unlabeled_action_gt.sum()}")
    print(f"Unlabeled Function GT (both function IDs = -1): {unlabeled_function_gt.sum()}")
    print(f"Invalid Action predictions (N/A): {invalid_action_pred.sum()}")
    print(f"Invalid Function predictions (N/A): {invalid_function_pred.sum()}")
    print(f"Total samples removed: {remove_mask.sum()}")
    
    merged_df = merged_df[~remove_mask].reset_index(drop=True)
    print(f"Valid samples for evaluation: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("ERROR: No valid samples remaining!")
        return
    
    # Evaluate Action Classification
    print(f"\n{'='*70}")
    print("ACTION CLASSIFICATION EVALUATION")
    print(f"{'='*70}")
    action_metrics = evaluate_single_task(
        merged_df, action_gt_cols, 'Action_Class', 'Action'
    )
    
    # Evaluate Function Classification
    print(f"\n{'='*70}")
    print("FUNCTION CLASSIFICATION EVALUATION")
    print(f"{'='*70}")
    function_metrics = evaluate_single_task(
        merged_df, function_gt_cols, 'Function_Class', 'Function'
    )
    
    # Combined Summary
    print(f"\n{'='*70}")
    print("COMBINED SUMMARY")
    print(f"{'='*70}")
    print(f"Valid samples evaluated: {len(merged_df)}")
    print(f"\n{'Action':<12} | Accuracy: {action_metrics['accuracy']:.4f} | "
          f"Macro-F1: {action_metrics['macro_f1']:.4f} | Micro-F1: {action_metrics['micro_f1']:.4f}")
    print(f"{'Function':<12} | Accuracy: {function_metrics['accuracy']:.4f} | "
          f"Macro-F1: {function_metrics['macro_f1']:.4f} | Micro-F1: {function_metrics['micro_f1']:.4f}")
    
    # Joint accuracy (both action AND function correct)
    joint_correct = 0
    for idx, row in merged_df.iterrows():
        action_gt = [row[c] for c in action_gt_cols if row[c] != -1]
        function_gt = [row[c] for c in function_gt_cols if row[c] != -1]
        
        action_match = row['Action_Class'] in action_gt
        function_match = row['Function_Class'] in function_gt
        
        if action_match and function_match:
            joint_correct += 1
    
    joint_accuracy = joint_correct / len(merged_df)
    print(f"\nJoint Accuracy (both correct): {joint_accuracy:.4f} ({joint_accuracy*100:.2f}%)")
    
    return merged_df, action_metrics, function_metrics


def evaluate_single_task(df, gt_cols, pred_col, task_name):
    """Evaluate a single classification task (Action or Function)."""
    
    y_true_multilabel = []
    y_pred_multilabel = []
    correct = 0
    
    for idx, row in df.iterrows():
        gt_labels = [row[c] for c in gt_cols if row[c] != -1]
        pred_label = row[pred_col]
        
        y_true_multilabel.append(gt_labels)
        y_pred_multilabel.append([pred_label] if pred_label != -1 else [])
        
        if pred_label in gt_labels:
            correct += 1
    
    accuracy = correct / len(df)
    print(f"\nSimple Accuracy (matches ANY GT): {correct}/{len(df)} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Get all unique IDs
    all_ids = set()
    for labels in y_true_multilabel:
        all_ids.update(labels)
    for labels in y_pred_multilabel:
        all_ids.update(labels)
    all_ids.discard(-1)
    all_ids = sorted(list(all_ids))
    
    print(f"Unique {task_name} IDs: {all_ids}")
    
    mlb = MultiLabelBinarizer(classes=all_ids)
    y_true_binary = mlb.fit_transform(y_true_multilabel)
    y_pred_binary = mlb.transform(y_pred_multilabel)
    
    # Classification report
    class_names = [str(i) for i in mlb.classes_]
    print(f"\nClassification Report:")
    print(classification_report(y_true_binary, y_pred_binary, target_names=class_names, 
                                zero_division=0, digits=4))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average=None, zero_division=0
    )
    
    metrics_df = pd.DataFrame({
        f'{task_name}_ID': mlb.classes_,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support.astype(int)
    })
    print(f"Per-{task_name} Metrics:")
    print(metrics_df.to_string(index=False))
    
    # Macro/Micro averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='macro', zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='micro', zero_division=0
    )
    
    print(f"\nMacro: P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f1:.4f}")
    print(f"Micro: P={micro_p:.4f}, R={micro_r:.4f}, F1={micro_f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1,
        'micro_precision': micro_p, 'micro_recall': micro_r, 'micro_f1': micro_f1,
        'metrics_df': metrics_df
    }


if __name__ == "__main__":
    results = evaluate_combined_classification(
        'ACL_LLM/new_csvs/ROSCO_1V_ID.csv',
        'ACL_LLM/llm_output/qwen3_dual_SCoT.csv'
    )
    
    if results is not None:
        merged_df, action_metrics, function_metrics = results
        print("\n✓ Evaluation completed successfully!")