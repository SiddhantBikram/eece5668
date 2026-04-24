import os
import argparse
import csv
from pathlib import Path
import re
from google import genai
from google.genai import types

# --- API Key Configuration ---
# Set your API key here or via environment variable GOOGLE_API_KEY
API_KEY = ''  # <-- PLACEHOLDER: Replace with your actual API key

# --- Few-Shot Example Configuration ---
# Map each class to an example video path and a brief reasoning
FEW_SHOT_EXAMPLES = {
    0: {
        "video_path": "Data/clips_filtered_full/TALK025_V1_05-52.8_05-55.8.mp4",  # <-- PLACEHOLDER
        "reasoning": "The child uses a vocalization to confirm that his mother did what he wanted.",
        "label": "Commenting"
    },
    1: {
        "video_path": "Data/clips_filtered_full/TALK001_V1_09-31.7_09-33.4.mp4",  # <-- PLACEHOLDER
        "reasoning": "The child expresses annoyance at her mom through a vocalization for making her use the AAC device.",
        "label": "Emotion"
    },
    2: {
        "video_path": "Data/clips_filtered_full/TALK014_V1_00-23.9_00-27.6.mp4",  # <-- PLACEHOLDER
        "reasoning": "The child pushes away an offered item and flails her hands while vocalizing.",
        "label": "Reject"
    },
    3: {
        "video_path": "Data/clips_filtered_full/TALK004b_V2_04-14.4_04-17.7.mp4",  # <-- PLACEHOLDER
        "reasoning": "The child hands her parent the ball, requesting him to open it.",
        "label": "Request"
    },
    4: {
        "video_path": "Data/clips_filtered_full/TALK010b_V1_10-29.0_10-33.5.mp4",  # <-- PLACEHOLDER
        "reasoning": "The child exhibits self-stimulatory behavior by brushing her fingers on the book.",
        "label": "Self-Directed Behavior"
    },
    5: {
        "video_path": "Data/clips_filtered_full/TALK033_V1_05-52.5_05-56.3.mp4",  # <-- PLACEHOLDER
        "reasoning": "The child is having fun playing with the toy and laughs with his mother, engaging in social reciprocity.",
        "label": "Social"
    },
}


def load_few_shot_examples(example_config: dict, fps: int) -> list[types.Part]:
    """Load few-shot example videos and create content parts."""
    parts = []
    
    for class_id in sorted(example_config.keys()):
        example = example_config[class_id]
        video_path = Path(example["video_path"])
        
        if not video_path.exists():
            raise FileNotFoundError(f"Few-shot example video not found: {video_path}")
        
        video_bytes = video_path.read_bytes()
        
        # Add example video
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    data=video_bytes,
                    mime_type='video/mp4'
                ),
                video_metadata=types.VideoMetadata(fps=fps)
            )
        )
        
        # Add example label with reasoning
        example_response = f"""Example classification for the video above:
{example['reasoning']} This is best classified as {class_id}: {example['label']}.

{class_id}"""
        parts.append(types.Part(text=example_response))
    
    return parts


def main():
    """
    Processes a directory of MP4 videos to classify a child's primary communicative
    function using Gemini 2.5 Flash with few-shot in-context learning.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Classify child actions in videos using Gemini 2.5 Flash with few-shot learning.'
    )
    parser.add_argument(
        '--input', '-i',
        default='Data/clips_filtered_full',
        help='Input directory containing MP4 videos'
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
    parser.add_argument(
        '--example-0',
        help='Path to example video for class 0 (Commenting)'
    )
    parser.add_argument(
        '--example-1',
        help='Path to example video for class 1 (Emotion)'
    )
    parser.add_argument(
        '--example-2',
        help='Path to example video for class 2 (Reject)'
    )
    parser.add_argument(
        '--example-3',
        help='Path to example video for class 3 (Request)'
    )
    parser.add_argument(
        '--example-4',
        help='Path to example video for class 4 (Self-Directed Behavior)'
    )
    parser.add_argument(
        '--example-5',
        help='Path to example video for class 5 (Social)'
    )
    args = parser.parse_args()

    # --- Update example paths from command line if provided ---
    example_config = FEW_SHOT_EXAMPLES.copy()
    for class_id in range(6):
        arg_value = getattr(args, f'example_{class_id}')
        if arg_value:
            example_config[class_id]["video_path"] = arg_value

    # --- Directory Validation ---
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"❌ Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'g3flash_icl_function.csv'

    # --- Read Existing CSV to Find Processed Videos ---
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

    # --- Filter Video List ---
    all_mp4_files = sorted(list(input_dir.glob('*.mp4')))
    files_to_process = [f for f in all_mp4_files if f.name not in processed_videos]

    if not files_to_process:
        print("✅ All videos are already processed. Nothing to do.")
        return

    print(f"Found {len(files_to_process)} new videos to process.")

    # --- Initialize Gemini Client ---
    print("🚀 Initializing Gemini client...")
    api_key = API_KEY if API_KEY != "YOUR_API_KEY_HERE" else os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: No API key provided. Set API_KEY in the script or GOOGLE_API_KEY environment variable.")
        return

    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini client initialized successfully!")
    except Exception as e:
        print(f"❌ Fatal Error: Could not initialize Gemini client. Exception: {e}")
        return

    # --- Load Few-Shot Examples ---
    print("📚 Loading few-shot example videos...")
    try:
        few_shot_parts = load_few_shot_examples(example_config, args.fps)
        print(f"✅ Loaded {len(example_config)} few-shot examples.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # --- Prompt Definition ---
    task_instruction = """Your task is to classify the primary communicative function of the child in this video. The child is non- or minimally-speaking and has profound autism and other complex neurodevelopmental disorders. Choose the most accurate option below.

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

Below are 6 example videos with their correct labels. Study these examples carefully, then classify the final video.
"""

    target_prompt = """
Carefully think through the answer by briefly detailing the particular movements that you see the child doing. Your output must contain your explanation, and then in a new line, a single integer corresponding to the option you choose. An example response is shown below: 

In the video, the child is pointing a single finger toward an object. This is most accurately described by option 3.

3"""

    # --- CSV Appending and Processing ---
    print(f"✍️ Appending new results to: {csv_path}")
    write_header = not csv_path.exists() or os.path.getsize(csv_path) == 0

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Video_Name', 'LLM_Reasoning', 'Final_Class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        # --- Main Processing Loop ---
        for i, video_path in enumerate(files_to_process, 1):
            reasoning = "Processing failed"
            final_class = "N/A"
            try:
                print(f"\n[{i}/{len(files_to_process)}] 🎬 Processing: {video_path.name}")

                # Read target video bytes
                video_bytes = video_path.read_bytes()
                video_size_mb = len(video_bytes) / (1024 * 1024)
                
                if video_size_mb > 20:
                    print(f"  > ⚠️ Warning: Video size ({video_size_mb:.1f}MB) exceeds 20MB inline limit.")

                # Build content: instruction + few-shot examples + target video + prompt
                all_parts = [
                    types.Part(text=task_instruction),
                    *few_shot_parts,
                    types.Part(text=target_prompt),
                    types.Part(
                        inline_data=types.Blob(
                            data=video_bytes,
                            mime_type='video/mp4'
                        ),
                        video_metadata=types.VideoMetadata(fps=args.fps)
                    ),
                ]

                content = types.Content(parts=all_parts)

                # Generate response
                response = client.models.generate_content(
                    model=args.model,
                    contents=content
                )

                # Extract response text
                response_text = response.text.strip()

                # --- Robust Parsing Logic ---
                match = re.search(r'^([0-5])\s*$', response_text, re.MULTILINE)
                if match:
                    final_class = match.group(1)
                    reasoning = response_text[:match.start()].strip()
                else:
                    reasoning = response_text
                    if response_text and response_text[-1] in '012345':
                        final_class = response_text[-1]

                print(f"  > Reasoning: {reasoning[:100]}...")
                print(f"  > Final Class: {final_class}")

            except Exception as e:
                reasoning = f"ERROR: {e}"
                print(f"  > ❌ An error occurred: {e}")

            finally:
                writer.writerow({
                    'Video_Name': video_path.name,
                    'LLM_Reasoning': reasoning,
                    'Final_Class': final_class
                })
                csvfile.flush()

    print(f"\n🎉 Processing complete! All results saved to: {csv_path}")


if __name__ == "__main__":
    main()