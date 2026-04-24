import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

def evaluate_action_classification(gt_file='ROSCO_GT.csv', pred_file='qwen3int.csv'):
    """
    Evaluate action/function classification by comparing predictions with ground truth.
    Now includes advanced metrics: MCC, Ambiguity Analysis, and Error Patterns.
    """
    
    # Load the CSV files
    print("Loading CSV files...")
    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)
    
    print(f"Ground truth samples: {len(gt_df)}")
    print(f"Prediction samples: {len(pred_df)}")
    
    # Merge dataframes on sample_name/Video_Name
    print("\nMatching samples between files...")
    merged_df = pd.merge(
        gt_df[['sample_name', 'FUNCTION1_ID', 'FUNCTION2_ID']], 
        pred_df[['Video_Name', 'Final_Class']], 
        left_on='sample_name', 
        right_on='Video_Name', 
        how='inner'
    )
    
    print(f"Matched samples: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("ERROR: No matching samples found between the two files!")
        print("\nSample of ground truth sample_name values:")
        print(gt_df['sample_name'].head())
        print("\nSample of prediction Video_Name values:")
        print(pred_df['Video_Name'].head())
        return
    
    # Convert action IDs to integers (handle NaN and float values)
    print("\nProcessing action IDs...")
    action_columns = ['FUNCTION1_ID', 'FUNCTION2_ID']
    
    for col in action_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        merged_df[col] = merged_df[col].fillna(-1).astype(int)
    
    merged_df['Final_Class'] = pd.to_numeric(merged_df['Final_Class'], errors='coerce')
    merged_df['Final_Class'] = merged_df['Final_Class'].fillna(-1).astype(int)
    
    # Filter out unlabeled samples (both FUNCTION1_ID and FUNCTION2_ID are -1)
    unlabeled_mask = (merged_df['FUNCTION1_ID'] == -1) & (merged_df['FUNCTION2_ID'] == -1)
    num_unlabeled = unlabeled_mask.sum()
    
    # Filter out samples with invalid predictions (-1, NaN, or N/A)
    invalid_pred_mask = merged_df['Final_Class'] == -1
    num_invalid_pred = invalid_pred_mask.sum()
    
    # Combined filter: remove if unlabeled OR invalid prediction
    remove_mask = unlabeled_mask | invalid_pred_mask
    
    print(f"\n{'='*60}")
    print(f"FILTERING SAMPLES:")
    print(f"{'='*60}")
    print(f"Total matched samples: {len(merged_df)}")
    print(f"Unlabeled GT samples (both labels = -1): {num_unlabeled}")
    print(f"Invalid prediction samples (N/A or missing): {num_invalid_pred}")
    print(f"Total samples removed: {remove_mask.sum()}")
    
    merged_df = merged_df[~remove_mask].reset_index(drop=True)
    print(f"Valid samples for evaluation: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("ERROR: No labeled samples remaining after filtering!")
        return
    
    # Create ground truth and prediction lists for multi-label classification
    y_true_multilabel = []
    y_pred_multilabel = []
    correct_predictions = 0
    
    # Track correctness per row for ambiguity analysis
    is_correct_list = []
    gt_counts = []
    
    for idx, row in merged_df.iterrows():
        gt_actions = []
        for col in action_columns:
            if row[col] != -1:
                gt_actions.append(row[col])
        
        pred_action = row['Final_Class']
        y_true_multilabel.append(gt_actions)
        
        # Store count of GT labels for this sample
        gt_counts.append(len(gt_actions))
        
        if pred_action != -1:
            y_pred_multilabel.append([pred_action])
        else:
            y_pred_multilabel.append([])
        
        # Check if prediction matches ANY ground truth
        if pred_action in gt_actions and pred_action != -1:
            correct_predictions += 1
            is_correct_list.append(True)
        else:
            is_correct_list.append(False)
    
    # Add helper columns for advanced analysis
    merged_df['is_correct'] = is_correct_list
    merged_df['gt_count'] = gt_counts

    simple_accuracy = correct_predictions / len(merged_df)
    
    print(f"\n{'='*60}")
    print(f"SIMPLE ACCURACY (Relaxed Top-1):")
    print(f"Correct predictions: {correct_predictions}/{len(merged_df)}")
    print(f"Accuracy: {simple_accuracy:.4f} ({simple_accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Multi-label classification report setup
    print("\nPreparing multi-label classification report...")
    
    all_actions = set()
    for actions in y_true_multilabel:
        all_actions.update(actions)
    for actions in y_pred_multilabel:
        all_actions.update(actions)
    
    all_actions.discard(-1)
    all_actions = sorted(list(all_actions))
    
    if len(all_actions) == 0:
        print("ERROR: No valid action IDs found in the data!")
        return
    
    print(f"Unique action IDs found: {all_actions}")
    
    mlb = MultiLabelBinarizer(classes=all_actions)
    y_true_binary = mlb.fit_transform(y_true_multilabel)
    y_pred_binary = mlb.transform(y_pred_multilabel)
    
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORT:")
    print(f"{'='*60}")
    
    class_names = [str(action) for action in mlb.classes_]
    report = classification_report(
        y_true_binary, y_pred_binary, target_names=class_names,
        zero_division=0, digits=4
    )
    print(report)
    
    # Calculate Per-Action Metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average=None, zero_division=0
    )
    
    metrics_df = pd.DataFrame({
        'Action_ID': mlb.classes_,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support.astype(int)
    })
    
    # --- NEW: ADVANCED METRICS SECTION ---
    print(f"\n{'='*60}")
    print("ADVANCED RESEARCH ANALYSIS:")
    print(f"{'='*60}")

    # 1. Ambiguity Analysis (Accuracy vs # of GT labels)
    print("\n[Analysis 1] Performance by Sample Ambiguity (Number of GT Labels):")
    ambiguity_df = merged_df.groupby('gt_count')['is_correct'].agg(['mean', 'count']).reset_index()
    ambiguity_df.columns = ['Num_GT_Labels', 'Accuracy', 'Sample_Count']
    print(ambiguity_df.to_string(index=False, formatters={'Accuracy': '{:.4f}'.format}))

    # 2. Matthews Correlation Coefficient (MCC)
    # Using FUNCTION1_ID vs Final_Class for a strict single-label proxy
    valid_mcc_rows = merged_df[merged_df['FUNCTION1_ID'] != -1]
    mcc_score = matthews_corrcoef(valid_mcc_rows['FUNCTION1_ID'], valid_mcc_rows['Final_Class'])
    print(f"\n[Analysis 2] Matthews Correlation Coefficient (GT=FUNCTION1): {mcc_score:.4f}")

    # 3. Class Imbalance Correlation
    corr = metrics_df['Support'].corr(metrics_df['F1-Score'])
    print(f"\n[Analysis 3] Correlation (Support vs F1): {corr:.4f}")
    if corr > 0.5:
        print("  -> High positive correlation: Model struggles significantly with rare classes.")
    elif corr < 0.1:
        print("  -> Low correlation: Model is relatively robust to class imbalance.")

    # 4. Top Confusions (Error Analysis)
    print("\n[Analysis 4] Top 5 Most Frequent Confusions (GT=FUNCTION1 vs Pred):")
    # Filter for errors only
    errors = valid_mcc_rows[valid_mcc_rows['FUNCTION1_ID'] != valid_mcc_rows['Final_Class']]
    if len(errors) > 0:
        error_counts = errors.groupby(['FUNCTION1_ID', 'Final_Class']).size().reset_index(name='Count')
        top_errors = error_counts.sort_values('Count', ascending=False).head(5)
        print(top_errors.to_string(index=False))
    else:
        print("  -> No errors found in valid samples.")

    # --- END ADVANCED METRICS ---

    print("\nPer-Action Metrics Summary:")
    print(metrics_df.to_string(index=False))
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='micro', zero_division=0
    )
    
    print(f"\n{'='*60}")
    print("OVERALL PERFORMANCE SUMMARY:")
    print(f"{'='*60}")
    print(f"Valid samples evaluated: {len(merged_df)}")
    print(f"Excluded - Unlabeled GT: {num_unlabeled}")
    print(f"Excluded - Invalid predictions: {num_invalid_pred}")
    print(f"Simple Accuracy (matches any GT action): {simple_accuracy:.4f}")
    print(f"MCC Score (Strict FUNCTION1 match): {mcc_score:.4f}")
    print(f"Macro-averaged Precision: {macro_precision:.4f}")
    print(f"Macro-averaged Recall: {macro_recall:.4f}")
    print(f"Macro-averaged F1-Score: {macro_f1:.4f}")
    print(f"Micro-averaged Precision: {micro_precision:.4f}")
    print(f"Micro-averaged Recall: {micro_recall:.4f}")
    print(f"Micro-averaged F1-Score: {micro_f1:.4f}")
    
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS (first 10 labeled samples):")
    print(f"{'='*60}")
    
    sample_df = merged_df.head(10).copy()
    sample_df['Match'] = sample_df.apply(
        lambda row: 'Yes' if row['Final_Class'] in [row['FUNCTION1_ID'], row['FUNCTION2_ID']] 
        and row['Final_Class'] != -1 else 'No', axis=1
    )
    print(sample_df[['sample_name', 'FUNCTION1_ID', 'FUNCTION2_ID', 'Final_Class', 'Match']].to_string(index=False))

    return merged_df, metrics_df

if __name__ == "__main__":
    results = evaluate_action_classification(
        # 'ACL_LLM/new_csvs/ROSCO_MV_ID.csv',
        'ACL_LLM/new_csvs/test.csv',
        'ACL_LLM/llm_output/g3flash_function.csv'
    )
    
    if results is not None:
        merged_df, metrics_df = results
        print("\n✓ Evaluation completed successfully!")