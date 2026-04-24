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


def main():
    """
    Processes a directory of MP4 videos to classify a child's primary communicative
    function using Gemini 2.5 Flash, skipping any videos already present in the output CSV.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Classify child actions in videos using Gemini 2.5 Flash, avoiding duplicates.'
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
    args = parser.parse_args()

    # --- Directory Validation ---
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"❌ Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'g3flash_action.csv'

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

    # --- Prompt Definition ---
    question = """
Your task is to classify the primary communicative action of the child in this video. The child is non- or minimally-speaking and has profound autism and other complex neurodevelopmental disorders. Choose the most accurate option below.

0: Alternate and Augmented Communication: Use of high-tech (tablets, speech-generating devices) or low-tech (picture cards, communication boards) AAC systems.
1: Body: Use of primarily the body or head. This includes more holistic movements like postural shifts (e.g., leaning in, turning away), or whole-body movements (e.g., rocking, walking away).
2: Face: Use of facial expressions to convey meaning (e.g., smiling, grimacing, frowning) that is not primarily a gaze shift.
3: Gesture: Use of the hands, arms, or limbs to communicate. This includes specific, directed movements like pointing, reaching, waving, or hand leading.
4: Looking: Use of eye gaze or head orientation to direct another person's attention to a specific subject, person, or location.
5: Vocalization: Use of any non-speech or speech-like sound made with the vocal tract to communicate (e.g., grunt, squeal, laugh, word approximation).

Reasoning Steps:
1.  First, identify the child in the video. You must ignore any adults present.
2.  Understand the most significant communicative action performed by the child.
3.  Choose the single best-fitting class from the list above.

Carefully think through the answer by briefly detailing the particular movements that you see the child doing. Your output must contain your explanation, and then in a new line, a single integer corresponding to the option you choose. An example response is shown below: 

In the video, the child is pointing a single finger toward an object. This is most accurately described by option 3.

3
"""

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

                # Read video bytes
                video_bytes = video_path.read_bytes()
                video_size_mb = len(video_bytes) / (1024 * 1024)
                
                if video_size_mb > 20:
                    print(f"  > ⚠️ Warning: Video size ({video_size_mb:.1f}MB) exceeds 20MB inline limit. Consider using File API for larger files.")

                # Create content with video and prompt
                content = types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=video_bytes,
                                mime_type='video/mp4'
                            ),
                            video_metadata=types.VideoMetadata(fps=args.fps)
                        ),
                        types.Part(text=question)
                    ]
                )

                # Generate response
                response = client.models.generate_content(
                    model=args.model,
                    contents=content
                )

                # Extract response text
                response_text = response.text.strip()

                # --- Robust Parsing Logic ---
                # Look for a single digit (0-5) on its own line at the end
                match = re.search(r'^([0-5])\s*$', response_text, re.MULTILINE)
                if match:
                    final_class = match.group(1)
                    # Everything before the final class is the reasoning
                    reasoning = response_text[:match.start()].strip()
                else:
                    # Fallback: check if last character is a valid class
                    reasoning = response_text
                    if response_text and response_text[-1] in '012345':
                        final_class = response_text[-1]

                print(f"  > Reasoning: {reasoning}")
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