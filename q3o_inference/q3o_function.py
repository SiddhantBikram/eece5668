import os

# SET THESE BEFORE IMPORTING SWIFT
os.environ['MODELSCOPE_CACHE'] = 'ACL_LLM/cache/Q3o/modelscope'
os.environ['HF_HOME'] = 'ACL_LLM/cache'
os.environ['HF_DATASETS_CACHE'] = 'ACL_LLM/cache'
os.environ['USE_AUDIO_IN_VIDEO'] = 'true'

import torch
import argparse
import csv
from pathlib import Path
import re
import pandas as pd

# MS-Swift imports (AFTER setting env vars)
from swift.llm import PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
from swift.tuners import Swift


def main():
    parser = argparse.ArgumentParser(
        description='Classify child communicative functions in videos using fine-tuned Qwen3 Omni LoRA.'
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
        '--cache-dir', '-c',
        default='ACL_LLM/cache',
        help='Cache directory for Hugging Face models'
    )
    parser.add_argument(
        '--model', '-m',
        default='Qwen/Qwen3-Omni-30B-A3B-Thinking',
        help='Base model path or HF model ID'
    )
    parser.add_argument(
        '--adapter', '-a',
        default='ACL_LLM/swift_output/q3o_shuffled_audio/checkpoint-1725',
        help='Path to LoRA adapter checkpoint'
    )
    parser.add_argument(
        '--test-csv',
        default='ACL_LLM/new_csvs/test.csv',
        help='Path to test CSV file with sample_name column'
    )
    parser.add_argument(
        '--output-name',
        default='q3o_finetuned_function_e5.csv',
        help='Name of the output CSV file'
    )
    parser.add_argument(
        '--use-audio',
        action='store_true',
        default=True,
        help='Use audio track in video processing'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=32,
        help='Maximum number of frames to sample from video'
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # --- Directory Validation ---
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"❌ Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.output_name

    # --- Load Test CSV to Get Target Videos ---
    test_csv_path = Path(args.test_csv)
    if not test_csv_path.exists():
        print(f"❌ Error: Test CSV '{test_csv_path}' does not exist.")
        return

    test_df = pd.read_csv(test_csv_path)
    if 'sample_name' not in test_df.columns:
        print(f"❌ Error: 'sample_name' column not found in test CSV.")
        print(f"   Available columns: {list(test_df.columns)}")
        return

    # Get target video names (add .mp4 if not present)
    target_videos = set()
    for name in test_df['sample_name'].dropna().unique():
        name = str(name)
        if not name.endswith('.mp4'):
            name = name + '.mp4'
        target_videos.add(name)

    print(f"🎯 Found {len(target_videos)} target videos in test CSV.")

    # --- Read Existing Output CSV to Find Processed Videos ---
    processed_videos = set()
    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Video_Name' in row:
                        processed_videos.add(row['Video_Name'])
            print(f"🔎 Found {len(processed_videos)} videos already processed.")
        except (IOError, csv.Error) as e:
            print(f"⚠️ Warning: Could not read existing CSV. Error: {e}")

    # --- Filter Video List ---
    files_to_process = []
    for video_name in target_videos:
        video_path = input_dir / video_name
        if video_path.exists() and video_name not in processed_videos:
            files_to_process.append(video_path)

    files_to_process = sorted(files_to_process)

    if not files_to_process:
        print("✅ All target videos are already processed. Nothing to do.")
        return

    print(f"📋 {len(files_to_process)} videos to process.")

    # --- Load Model with MS-Swift ---
    print("🚀 Loading base model...")

    try:
        # Step 1: Load base model and tokenizer
        model, tokenizer = get_model_tokenizer(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_impl='flash_attention_2',
        )
        print("✅ Base model loaded!")
        
        # Debug: Check model type
        print(f"   Model type: {type(model).__name__}")

        # Step 2: Load LoRA adapter
        print(f"🔧 Loading LoRA adapter from: {args.adapter}")
        model = Swift.from_pretrained(model, args.adapter)
        model.eval()
        print("✅ LoRA adapter loaded!")

        # Step 3: Get template and create engine
        # Try different template names for Qwen3-Omni
        template_names = ['qwen3-omni', 'qwen3_omni', 'qwen2_5_omni']
        template = None
        
        for tname in template_names:
            try:
                template = get_template(tname, tokenizer)
                print(f"✅ Using template: {tname}")
                break
            except Exception as e:
                print(f"   Template '{tname}' failed: {e}")
                continue
        
        if template is None:
            print("❌ Could not find valid template. Trying default...")
            template = get_template(tokenizer=tokenizer)
        
        engine = PtEngine.from_model_template(model, template)
        print("✅ Inference engine ready!")

    except Exception as e:
        print(f"❌ Fatal Error: Could not load model. Exception: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Prompt Definition (Function Classification) ---
    question = """Your task is to classify the primary communicative function of the child in this video. The child is non- or minimally-speaking and has profound autism and other complex neurodevelopmental disorders. Choose the most accurate option below.

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

Carefully think through the answer by briefly detailing the particular movements that you see the child doing. Your output must contain your explanation, and then in a new line, a single integer corresponding to the option you choose. An example response is shown below: 

In the video, the child is pointing a single finger toward an object. This is most accurately described by option 3.

3"""

    # --- Request config ---
    request_config = RequestConfig(max_tokens=512, temperature=0)

    # --- CSV Appending and Processing ---
    print(f"✍️ Writing results to: {csv_path}")
    write_header = not csv_path.exists() or os.path.getsize(csv_path) == 0

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Video_Name', 'LLM_Reasoning', 'Final_Class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for i, video_path in enumerate(files_to_process, 1):
            reasoning = "Processing failed"
            final_class = "N/A"
            try:
                print(f"\n[{i}/{len(files_to_process)}] 🎬 Processing: {video_path.name}")

                # === FIXED: Use structured content format for video ===
                # Method 1: Using content list with typed dictionaries
                infer_request = InferRequest(
                    messages=[
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'video',
                                    'video': str(video_path),
                                },
                                {
                                    'type': 'text',
                                    'text': question
                                }
                            ]
                        }
                    ]
                )

                # Run inference
                resp_list = engine.infer([infer_request], request_config)
                response_text = resp_list[0].choices[0].message.content.strip()

                # --- Parsing Logic ---
                # Remove <think>...</think> blocks if present (for thinking models)
                clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                
                # Try to find the final class number
                match = re.search(r'^([0-5])\s*$', clean_response, re.MULTILINE)
                if match:
                    final_class = match.group(1)
                    reasoning = clean_response[:match.start()].strip()
                else:
                    reasoning = clean_response
                    # Look for last digit 0-5 in the response
                    digit_matches = re.findall(r'[0-5]', clean_response)
                    if digit_matches:
                        final_class = digit_matches[-1]

                print(f"  > Reasoning: {reasoning[:100]}...")
                print(f"  > Final Class: {final_class}")

            except Exception as e:
                reasoning = f"ERROR: {e}"
                print(f"  > ❌ An error occurred: {e}")
                import traceback
                traceback.print_exc()

            finally:
                writer.writerow({
                    'Video_Name': video_path.name,
                    'LLM_Reasoning': reasoning,
                    'Final_Class': final_class
                })
                csvfile.flush()

    print(f"\n🎉 Processing complete! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()