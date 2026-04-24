import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

def evaluate_action_classification(gt_file='ROSCO_GT.csv', pred_file='qwen3int.csv'):
    """
    Evaluate action classification with participant-wise metrics.
    Participants are identified by unique values in the 'TALK' column.
    """
    
    print("Loading CSV files...")
    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)
    
    print(f"Ground truth samples: {len(gt_df)}")
    print(f"Prediction samples: {len(pred_df)}")
    
    # Merge dataframes - include TALK column
    print("\nMatching samples between files...")
    merged_df = pd.merge(
        gt_df[['sample_name', 'TALK', 'ACTION1_ID', 'ACTION2_ID', 'ACTION3_ID']], 
        pred_df[['Video_Name', 'Final_Class']], 
        left_on='sample_name', 
        right_on='Video_Name', 
        how='inner'
    )
    
    print(f"Matched samples: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("ERROR: No matching samples found!")
        return
    
    # Convert action IDs to integers
    print("\nProcessing action IDs...")
    action_columns = ['ACTION1_ID', 'ACTION2_ID', 'ACTION3_ID']

    for col in action_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(-1).astype(int)
    
    merged_df['Final_Class'] = pd.to_numeric(merged_df['Final_Class'], errors='coerce').fillna(-1).astype(int)
    
    # Filter out unlabeled and invalid samples
    unlabeled_mask = (
        (merged_df['ACTION1_ID'] == -1) & 
        (merged_df['ACTION2_ID'] == -1) & 
        (merged_df['ACTION3_ID'] == -1)
    )
    invalid_pred_mask = merged_df['Final_Class'] == -1
    remove_mask = unlabeled_mask | invalid_pred_mask
    
    print(f"\n{'='*70}")
    print("FILTERING SAMPLES:")
    print(f"{'='*70}")
    print(f"Total matched samples: {len(merged_df)}")
    print(f"Unlabeled GT samples: {unlabeled_mask.sum()}")
    print(f"Invalid prediction samples: {invalid_pred_mask.sum()}")
    print(f"Total samples removed: {remove_mask.sum()}")
    
    merged_df = merged_df[~remove_mask].reset_index(drop=True)
    print(f"Valid samples for evaluation: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("ERROR: No valid samples remaining!")
        return
    
    # Get unique participants
    participants = sorted(merged_df['TALK'].unique())
    print(f"\nUnique participants (TALK): {len(participants)}")
    
    # Get all unique action IDs across entire dataset for consistent MLB
    all_actions = set()
    for col in action_columns:
        all_actions.update(merged_df[col].unique())
    all_actions.update(merged_df['Final_Class'].unique())
    all_actions.discard(-1)
    all_actions = sorted(list(all_actions))
    
    print(f"Unique action IDs: {all_actions}")
    
    # Initialize MLB with all classes
    mlb = MultiLabelBinarizer(classes=all_actions)
    mlb.fit([all_actions])  # Fit with all classes
    
    # Store participant-wise metrics
    participant_metrics = []
    
    print(f"\n{'='*70}")
    print("PARTICIPANT-WISE EVALUATION")
    print(f"{'='*70}")
    
    for participant in participants:
        p_df = merged_df[merged_df['TALK'] == participant].copy()
        
        if len(p_df) == 0:
            continue
        
        y_true_multilabel = []
        y_pred_multilabel = []
        correct = 0
        
        for idx, row in p_df.iterrows():
            gt_actions = [row[col] for col in action_columns if row[col] != -1]
            pred_action = row['Final_Class']
            
            y_true_multilabel.append(gt_actions)
            y_pred_multilabel.append([pred_action] if pred_action != -1 else [])
            
            if pred_action in gt_actions:
                correct += 1
        
        accuracy = correct / len(p_df)
        
        # Binarize
        y_true_binary = mlb.transform(y_true_multilabel)
        y_pred_binary = mlb.transform(y_pred_multilabel)
        
        # Macro metrics
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='macro', zero_division=0
        )
        
        # Micro metrics
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='micro', zero_division=0
        )
        
        participant_metrics.append({
            'Participant': participant,
            'Samples': len(p_df),
            'Accuracy': accuracy,
            'Macro_F1': macro_f1,
            'Macro_Precision': macro_p,
            'Macro_Recall': macro_r,
            'Micro_Precision': micro_p,
            'Micro_Recall': micro_r,
            'Micro_F1': micro_f1,

        })
    
    # Create summary dataframe
    metrics_df = pd.DataFrame(participant_metrics)
    
    print("\nParticipant-wise Metrics Summary:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Overall metrics (all samples)
    print(f"\n{'='*70}")
    print("OVERALL METRICS (All Participants)")
    print(f"{'='*70}")
    
    y_true_all = []
    y_pred_all = []
    total_correct = 0
    
    for idx, row in merged_df.iterrows():
        gt_actions = [row[col] for col in action_columns if row[col] != -1]
        pred_action = row['Final_Class']
        
        y_true_all.append(gt_actions)
        y_pred_all.append([pred_action] if pred_action != -1 else [])
        
        if pred_action in gt_actions:
            total_correct += 1
    
    overall_accuracy = total_correct / len(merged_df)
    
    y_true_binary_all = mlb.transform(y_true_all)
    y_pred_binary_all = mlb.transform(y_pred_all)
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_binary_all, y_pred_binary_all, average='macro', zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_binary_all, y_pred_binary_all, average='micro', zero_division=0
    )
    
    print(f"Total samples: {len(merged_df)}")
    print(f"Simple Accuracy: {overall_accuracy:.4f}")
    print(f"Micro - P: {micro_p:.4f}, R: {micro_r:.4f}, F1: {micro_f1:.4f}")
    print(f"Macro - P: {macro_p:.4f}, R: {macro_r:.4f}, F1: {macro_f1:.4f}")
    
    # Participant averages
    print(f"\n{'='*70}")
    print("PARTICIPANT-AVERAGED METRICS")
    print(f"{'='*70}")
    print(f"Mean Accuracy: {metrics_df['Accuracy'].mean():.4f} ± {metrics_df['Accuracy'].std():.4f}")
    print(f"Mean Micro F1: {metrics_df['Micro_F1'].mean():.4f} ± {metrics_df['Micro_F1'].std():.4f}")
    print(f"Mean Macro Precision: {metrics_df['Macro_Precision'].mean():.4f} ± {metrics_df['Macro_Precision'].std():.4f}")
    print(f"Mean Macro Recall: {metrics_df['Macro_Recall'].mean():.4f} ± {metrics_df['Macro_Recall'].std():.4f}")
    print(f"Mean Macro F1: {metrics_df['Macro_F1'].mean():.4f} ± {metrics_df['Macro_F1'].std():.4f}")
    
    return merged_df, metrics_df


if __name__ == "__main__":
    results = evaluate_action_classification(
        'ACL_LLM/new_csvs/test.csv',
        'ACL_LLM/llm_output/q3o_finetuned_think_e3.csv'
    )
    
    if results is not None:
        merged_df, metrics_df = results
        print("\n✓ Evaluation completed successfully!")
        
        # Optional: save participant metrics
        # metrics_df.to_csv('participant_metrics.csv', index=False)