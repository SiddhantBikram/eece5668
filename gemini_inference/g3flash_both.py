import os
import argparse
import csv
from pathlib import Path
import re
import pandas as pd
from google import genai
from google.genai import types

# --- API Key Configuration ---
API_KEY = ''


def main():
    """
    Processes a directory of MP4 videos to classify both the child's primary action
    and communicative function using Gemini.
    """
    parser = argparse.ArgumentParser(
        description='Classify child actions and functions in videos using Gemini.'
    )
    parser.add_argument(
        '--input', '-i',
        default='Data/clips_filtered_full',
        help='Input directory containing MP4 videos'
    )
    parser.add_argument(
        '--gt-csv', '-g',
        default='ACL_LLM/ROSCO_GT.csv',
        help='Ground truth CSV to filter which videos to process'
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

    # Directory Validation
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
    csv_path = output_dir / 'g3flash_both.csv'

    # Load ground truth CSV and create mapping
    print(f"📄 Loading ground truth CSV: {gt_csv_path}")
    gt_df = pd.read_csv(gt_csv_path)
    
    # Determine the video name column
    video_col = None
    for col in ['Video_Name', 'video_name', 'filename', 'Filename', 'sample_name']:
        if col in gt_df.columns:
            video_col = col
            break
    
    if video_col is None:
        print(f"⚠️ Warning: Could not find video name column. Available columns: {list(gt_df.columns)}")
        print("Please specify which column contains video names.")
        return
    
    # Create a set of valid video names from GT
    gt_videos = set(gt_df[video_col].tolist())
    print(f"✅ Loaded {len(gt_videos)} entries from ground truth CSV.")

    # Read Existing CSV to Find Processed Videos
    processed_videos = set()
    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Video_Name' in row:
                        processed_videos.add(row['Video_Name'])
            print(f"🔎 Found {len(processed_videos)} videos already processed in '{csv_path}'.")
        except (IOError, csv.Error) as e:
            print(f"⚠️ Warning: Could not read existing CSV file. Will create a new one. Error: {e}")

    # Filter Video List - only process videos that are in the GT CSV
    all_mp4_files = sorted(list(input_dir.glob('*.mp4')))
    files_to_process = [
        f for f in all_mp4_files 
        if f.name not in processed_videos and f.name in gt_videos
    ]

    # Report videos not in GT
    videos_not_in_gt = [f.name for f in all_mp4_files if f.name not in gt_videos]
    if videos_not_in_gt:
        print(f"⚠️ {len(videos_not_in_gt)} videos not found in GT CSV (will be skipped)")

    if not files_to_process:
        print("✅ All videos are already processed or not in GT. Nothing to do.")
        return

    print(f"Found {len(files_to_process)} new videos to process.")

    # Initialize Gemini Client
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

    # Prompt for dual classification (same as Qwen)
    question = """
Your task is to classify the primary communicative action and function of the child in this video. The child is non- or minimally-speaking and has profound autism and other complex neurodevelopmental disorders. Choose the most accurate option below.

ACTION Class Definitions:
 0: Alternate and Augmented Communication: Use of high-tech (tablets, speech-generating devices) or low-tech (picture cards, communication boards) AAC systems.
 1: Body: Use of primarily the body or head. This includes more holistic movements like postural shifts (e.g., leaning in, turning away), or whole-body movements (e.g., rocking, walking away).
 2: Face: Use of facial expressions to convey meaning (e.g., smiling, grimacing, frowning) that is not primarily a gaze shift.
 3: Gesture: Use of the hands, arms, or limbs to communicate. This includes specific, directed movements like pointing, reaching, waving, or hand leading.
 4: Looking: Use of eye gaze or head orientation to direct another person's attention to a specific subject, person, or location.
 5: Vocalization: Use of any non-speech or speech-like sound made with the vocal tract to communicate (e.g., grunt, squeal, laugh, word approximation).

FUNCTION Class Definitions:
 0: Commenting: Sharing observations, expressing interest, or drawing attention to objects or events.
 1: Emotion: Expressing an internal feeling state like happiness, frustration, excitement, or distress.
 2: Reject: Refusing items, activities, pushing away, head shaking, or vocal protests indicating a desire for an activity to stop.
 3: Request: Asking for objects, actions, continuation, or assistance through any modality.
 4: Self-Directed Behavior: Behaviors that were not perceived as being intentionally communicative or directed at another person, like self-stimulatory behavior.
 5: Social: Initiating social interaction, maintaining social reciprocity, greeting, responding to a name, or joint attention.

Reasoning Steps:
1. First, identify the child in the video. You must ignore any adults present.
2. Determine the primary action (how the child is communicating).
3. Determine the primary function (why the child is communicating).
4. Choose the single best-fitting class for each category.

Carefully think through the answer by briefly detailing the particular movements that you see the child doing. After your reasoning, your response MUST end with exactly this format on the last two lines:
Action: <single digit 0-5>
Function: <single digit 0-5>
"""

    # CSV Appending and Processing
    print(f"✍️ Appending new results to: {csv_path}")
    write_header = not csv_path.exists() or os.path.getsize(csv_path) == 0

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Video_Name', 'LLM_Reasoning', 'Action_Class', 'Function_Class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for i, video_path in enumerate(files_to_process, 1):
            reasoning = "Processing failed"
            action_class = "N/A"
            function_class = "N/A"
            
            try:
                print(f"\n[{i}/{len(files_to_process)}] 🎬 Processing: {video_path.name}")

                # Read video bytes
                video_bytes = video_path.read_bytes()

                # Create content
                content = types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                            video_metadata=types.VideoMetadata(fps=args.fps)
                        ),
                        types.Part(text=question)
                    ]
                )

                # Generate response with retry logic for PROHIBITED_CONTENT
                max_retries = 5
                for attempt in range(1, max_retries + 1):
                    response = client.models.generate_content(model=args.model, contents=content)
                    
                    # Handle None response
                    if response.text is not None:
                        reasoning = response.text.strip()
                        
                        # Robust Parsing Logic for dual output (same as Qwen)
                        action_match = re.search(r'Action:\s*(\d)', reasoning, re.IGNORECASE)
                        function_match = re.search(r'Function:\s*(\d)', reasoning, re.IGNORECASE)
                        
                        if action_match:
                            action_class = action_match.group(1)
                        if function_match:
                            function_class = function_match.group(1)
                        
                        # Fallback: look for last two digits if structured format not found
                        if action_class == "N/A" or function_class == "N/A":
                            digits = re.findall(r'\b([0-5])\b', reasoning[-100:])
                            if len(digits) >= 2:
                                if action_class == "N/A":
                                    action_class = digits[-2]
                                if function_class == "N/A":
                                    function_class = digits[-1]
                            elif len(digits) == 1:
                                if action_class == "N/A":
                                    action_class = digits[0]

                        print(f"  > Reasoning: {reasoning[:100]}...")
                        print(f"  > Action Class: {action_class}")
                        print(f"  > Function Class: {function_class}")
                        break
                    else:
                        # Check if blocked by safety filters
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                            reasoning = f"BLOCKED: {response.prompt_feedback}"
                        elif hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'finish_reason'):
                                finish_reason = str(candidate.finish_reason)
                                reasoning = f"NO_TEXT: finish_reason={candidate.finish_reason}"
                                
                                # Retry if PROHIBITED_CONTENT
                                if 'PROHIBITED_CONTENT' in finish_reason:
                                    if attempt < max_retries:
                                        print(f"  > ⚠️ Attempt {attempt}/{max_retries}: PROHIBITED_CONTENT, retrying...")
                                        continue
                                    else:
                                        print(f"  > ❌ Attempt {attempt}/{max_retries}: PROHIBITED_CONTENT, giving up.")
                            else:
                                reasoning = "NO_TEXT: Empty response"
                        else:
                            reasoning = "NO_TEXT: Response was None"
                        
                        print(f"  > ⚠️ {reasoning}")
                        break

            except Exception as e:
                reasoning = f"ERROR: {e}"
                print(f"  > ❌ An error occurred: {e}")

            finally:
                writer.writerow({
                    'Video_Name': video_path.name,
                    'LLM_Reasoning': reasoning,
                    'Action_Class': action_class,
                    'Function_Class': function_class
                })
                csvfile.flush()

    print(f"\n🎉 Processing complete! All results saved to: {csv_path}")


if __name__ == "__main__":
    main()