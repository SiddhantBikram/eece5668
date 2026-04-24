import os
import argparse
import csv
from pathlib import Path
import pandas as pd
from google import genai
from google.genai import types

# --- API Key Configuration ---
API_KEY = ''

# Function ID to text mapping
FUNCTION_MAP = {
    0: "Commenting: Sharing observations, expressing interest, or drawing attention to objects or events.",
    1: "Emotion: Expressing an internal feeling state like happiness, frustration, excitement, or distress.",
    2: "Reject: Refusing items, activities, pushing away, head shaking, or vocal protests indicating a desire for an activity to stop.",
    3: "Request: Asking for objects, actions, continuation, or assistance through any modality.",
    4: "Self-Directed Behavior: Behaviors that were not perceived as being intentionally communicative or directed at another person, like self-stimulatory behavior.",
    5: "Social: Initiating social interaction, maintaining social reciprocity, greeting, responding to a name, or joint attention."
}

def is_valid_value(val):
    """Check if a value is valid (not NaN, not 'N/A', not empty string)."""
    if pd.isna(val):
        return False
    if isinstance(val, str) and val.strip().upper() in ['N/A', 'NA', '']:
        return False
    return True

def has_any_function(row):
    """Check if the row has at least one valid function label."""
    for col in ['FUNCTION1_ID', 'FUNCTION2_ID']:
        if is_valid_value(row.get(col)):
            return True
    return False

def get_function_hint(row):
    """Generate function hint text based on FUNCTION1_ID and FUNCTION2_ID."""
    functions = []
    
    for col in ['FUNCTION1_ID', 'FUNCTION2_ID']:
        val = row.get(col)
        if is_valid_value(val):
            func_id = int(val)
            func_text = FUNCTION_MAP.get(func_id, f"Unknown({func_id})")
            functions.append(func_text)
    
    if len(functions) == 0:
        return ""
    elif len(functions) == 1:
        return f"Hint: The child's primary communicative function in this video is '{functions[0]}'."
    else:
        return f"Hint: The child's communicative functions in this video are '{functions[0]}' and '{functions[1]}'."

def is_failed_description(desc):
    """Check if a description indicates a failed processing attempt."""
    if not desc:
        return True
    failed_markers = ['NO_TEXT', 'BLOCKED', 'Processing failed', 'ERROR:']
    return any(marker in desc for marker in failed_markers)


def main():
    """
    Generates detailed function descriptions using label-guided reasoning (hints).
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Generate function captions with hints using Gemini (CARE-VL style).'
    )
    parser.add_argument(
        '--input', '-i',
        default='Data/clips_filtered_full',
        help='Input directory containing MP4 videos'
    )
    parser.add_argument(
        '--gt-csv', '-g',
        default='ACL_LLM/new_csvs/ROSCO_1V_ID.csv',
        help='Ground truth CSV containing FUNCTION1_ID and FUNCTION2_ID'
    )
    parser.add_argument(
        '--output', '-o',
        default='ACL_LLM/llm_output',
        help='Output directory for the resulting CSV file'
    )
    parser.add_argument(
        '--model', '-m',
        default='models/gemini-3-flash-preview',
        help='Gemini model to use'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=2,
        help='Frames per second to sample from video'
    )
    args = parser.parse_args()

    # --- Directory Validation ---
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    gt_csv_path = Path(args.gt_csv)

    if not input_dir.is_dir():
        print(f"❌ Error: Input directory '{input_dir}' does not exist.")
        return

    if not gt_csv_path.exists():
        print(f"❌ Error: Ground truth CSV '{gt_csv_path}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'g3flash_function_qa.csv'

    # --- Load ground truth CSV ---
    print(f"📄 Loading ground truth CSV: {gt_csv_path}")
    gt_df = pd.read_csv(gt_csv_path)
    
    video_col = None
    for col in ['Video_Name', 'video_name', 'filename', 'Filename', 'sample_name']:
        if col in gt_df.columns:
            video_col = col
            break
    
    if video_col is None:
        print(f"⚠️ Could not find video name column. Available: {list(gt_df.columns)}")
        return
    
    # Only include rows that have at least one valid function label
    gt_mapping = {}
    skipped_no_function = 0
    for _, row in gt_df.iterrows():
        video_name = row[video_col]
        if has_any_function(row):
            gt_mapping[video_name] = row
        else:
            skipped_no_function += 1
    
    print(f"✅ Loaded {len(gt_mapping)} entries with valid function labels from ground truth CSV.")
    if skipped_no_function > 0:
        print(f"⏭️ Skipped {skipped_no_function} entries with no function labels.")

    # --- Read Existing CSV ---
    processed_videos = set()
    failed_videos = set()
    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Video_Name' in row:
                        desc = row.get('Function_Description', '')
                        if is_failed_description(desc):
                            failed_videos.add(row['Video_Name'])
                        else:
                            processed_videos.add(row['Video_Name'])
            print(f"🔎 Found {len(processed_videos)} videos successfully processed.")
            if failed_videos:
                print(f"🔄 Found {len(failed_videos)} failed videos to retry.")
        except (IOError, csv.Error) as e:
            print(f"⚠️ Warning: Could not read existing CSV. Error: {e}")

    # --- Filter Video List ---
    all_mp4_files = sorted(list(input_dir.glob('*.mp4')))
    files_to_process = [
        f for f in all_mp4_files 
        if f.name not in processed_videos and f.name in gt_mapping
    ]

    if not files_to_process:
        print("✅ All videos are already processed or not in GT. Nothing to do.")
        return

    new_count = len([f for f in files_to_process if f.name not in failed_videos])
    retry_count = len([f for f in files_to_process if f.name in failed_videos])
    print(f"Found {new_count} new videos and {retry_count} retries to process.")

    # --- Remove failed entries from CSV before retrying ---
    if failed_videos and csv_path.exists():
        print("🧹 Removing failed entries from CSV for retry...")
        temp_path = csv_path.with_suffix('.tmp')
        with open(csv_path, 'r', newline='', encoding='utf-8') as infile, \
             open(temp_path, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for row in reader:
                if row.get('Video_Name') not in failed_videos:
                    writer.writerow(row)
        temp_path.replace(csv_path)
        print(f"✅ Removed {len(failed_videos)} failed entries.")

    # --- Initialize Gemini Client ---
    print("🚀 Initializing Gemini client...")
    api_key = API_KEY if API_KEY != "YOUR_API_KEY_HERE" else os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: No API key provided.")
        return

    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini client initialized successfully!")
    except Exception as e:
        print(f"❌ Fatal Error: Could not initialize Gemini client. Exception: {e}")
        return

    # --- Prompt Template with Hint ---
    prompt_template = """
Your task is to classify the primary communicative function of the child in this video. The child is non- or minimally-speaking and has profound autism and other complex neurodevelopmental disorders. Choose the most accurate option below.

0: Commenting: Sharing observations, expressing interest, or drawing attention to objects or events.
1: Emotion: Expressing an internal feeling state like happiness, frustration, excitement, or distress.
2: Reject: Refusing items, activities, pushing away, head shaking, or vocal protests indicating a desire for an activity to stop.
3: Request: Asking for objects, actions, continuation, or assistance through any modality.
4: Self-Directed Behavior: Behaviors that were not perceived as being intentionally communicative or directed at another person, like self-stimulatory behavior.
5: Social: Initiating social interaction, maintaining social reciprocity, greeting, responding to a name, or joint attention.

Reasoning Steps:
1.  First, identify the child in the video. You must ignore any adults present.
2.  Understand the most significant communicative function performed by the child.
3.  Choose the single best-fitting class from the list above.

Carefully think through the answer by briefly detailing the particular movements that you see the child doing. 

{hint}

Your output must contain your detailed explanation, and then in a new line, a single integer corresponding to the option you choose. An example response is shown below: 

In the video, the child is pointing a single finger toward an object. This is most accurately described by option 3.

3
"""

    # --- CSV Processing ---
    print(f"✍️ Appending new results to: {csv_path}")
    write_header = not csv_path.exists() or os.path.getsize(csv_path) == 0

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Video_Name', 'Function_Hint', 'Function_Description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        if write_header:
            writer.writeheader()

        for i, video_path in enumerate(files_to_process, 1):
            description = "Processing failed"
            hint = ""
            
            try:
                print(f"\n[{i}/{len(files_to_process)}] 🎬 Processing: {video_path.name}")

                # Get hint from GT
                gt_row = gt_mapping[video_path.name]
                hint = get_function_hint(gt_row)
                
                if hint:
                    print(f"  > Hint: {hint[:80]}...")
                else:
                    # This shouldn't happen since we filtered earlier, but just in case
                    print("  > No hint available, skipping...")
                    continue

                # Build prompt with hint
                prompt = prompt_template.format(hint=hint)

                # Read video bytes
                video_bytes = video_path.read_bytes()

                # Create content
                content = types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                            video_metadata=types.VideoMetadata(fps=args.fps)
                        ),
                        types.Part(text=prompt)
                    ]
                )

                # Generate response with retry logic for PROHIBITED_CONTENT
                max_retries = 5
                for attempt in range(1, max_retries + 1):
                    response = client.models.generate_content(model=args.model, contents=content)
                    
                    # Handle None response
                    if response.text is not None:
                        description = response.text.strip()
                        print(f"  > Description: {description[:120]}...")
                        break
                    else:
                        # Check if blocked by safety filters
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                            description = f"BLOCKED: {response.prompt_feedback}"
                        elif hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'finish_reason'):
                                finish_reason = str(candidate.finish_reason)
                                description = f"NO_TEXT: finish_reason={candidate.finish_reason}"
                                
                                # Retry if PROHIBITED_CONTENT
                                if 'PROHIBITED_CONTENT' in finish_reason:
                                    if attempt < max_retries:
                                        print(f"  > ⚠️ Attempt {attempt}/{max_retries}: PROHIBITED_CONTENT, retrying...")
                                        continue
                                    else:
                                        print(f"  > ❌ Attempt {attempt}/{max_retries}: PROHIBITED_CONTENT, giving up.")
                            else:
                                description = "NO_TEXT: Empty response"
                        else:
                            description = "NO_TEXT: Response was None"
                        
                        print(f"  > ⚠️ {description}")
                        break

            except Exception as e:
                description = f"ERROR: {e}"
                print(f"  > ❌ An error occurred: {e}")

            finally:
                writer.writerow({
                    'Video_Name': video_path.name,
                    'Function_Hint': hint,
                    'Function_Description': description
                })
                csvfile.flush()

    print(f"\n🎉 Processing complete! All results saved to: {csv_path}")


if __name__ == "__main__":
    main()