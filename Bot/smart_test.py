#!/usr/bin/env python3
"""
Smart GymBot Analyzer - Auto-detects images and videos
Analyzes any combination of photos and videos
"""

import sys
import os

print("="*60)
print("🤖 GymBot Smart Analyzer")
print("="*60)

if len(sys.argv) < 2:
    print("\nUsage:")
    print("  python smart_test.py <file1> [file2] [file3] ...")
    print("\nExamples:")
    print("  python smart_test.py squat_form.jpg")
    print("  python smart_test.py workout_set.mov")
    print("  python smart_test.py video.mov photo1.jpg photo2.jpg")
    print("  python smart_test.py front.jpg side.jpg back.jpg")
    print("\nSupported formats:")
    print("  Images: .jpg .jpeg .png .webp .gif .bmp")
    print("  Videos: .mp4 .mov .avi .mkv .webm")
    sys.exit(1)

file_paths = sys.argv[1:]

# Verify all files exist
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

# Detect file types
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}

images = []
videos = []

for file_path in file_paths:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTS:
        images.append(file_path)
    elif ext in VIDEO_EXTS:
        videos.append(file_path)
    else:
        print(f"⚠️  Unknown file type: {file_path} (extension: {ext})")
        print("Skipping this file...")

if not images and not videos:
    print("❌ No valid image or video files found!")
    sys.exit(1)

# Display what we found
print(f"\n📁 Detected:")
if videos:
    print(f"   📹 {len(videos)} video(s):")
    for i, vid in enumerate(videos, 1):
        print(f"      {i}. {os.path.basename(vid)}")
if images:
    print(f"   📸 {len(images)} image(s):")
    for i, img in enumerate(images, 1):
        print(f"      {i}. {os.path.basename(img)}")

print("\n" + "-"*60)

try:
    from Spotter_AI import chat_vision, load_image, load_video
    
    all_images = []
    
    # Load videos (extract frames)
    if videos:
        print("🎬 Processing videos...")
        for vid_path in videos:
            print(f"   Extracting frames from {os.path.basename(vid_path)}...")
            frames = load_video(vid_path, max_frames=8)
            all_images.extend(frames)
            print(f"   ✅ Extracted {len(frames)} frames")
    
    # Load images
    if images:
        print("📸 Loading images...")
        for img_path in images:
            img = load_image(img_path)
            all_images.append(img)
            print(f"   ✅ Loaded {os.path.basename(img_path)} ({img.size[0]}x{img.size[1]})")
    
    print(f"\n📊 Total images to analyze: {len(all_images)}")
    
    # Build smart prompt based on what we have
    if videos and images:
        # Mix of videos and images
        video_frame_count = sum(8 for _ in videos)  # 8 frames per video
        prompt = f"""You are analyzing {len(videos)} workout video(s) plus {len(images)} photo(s).

**Video frames (first {video_frame_count} images):** Show complete exercise sets and movement patterns

**Additional photos (next {len(images)} images):** Different angles, positions, or details

**Analysis requested:**

1. **Exercise Identification**: What exercise(s) are shown?

2. **Video Analysis**:
   - Form quality across the set(s) (1-10)
   - Consistency across reps
   - Movement patterns and tempo
   - Where does form break down?

3. **Photo Analysis**:
   - What do the additional photos show?
   - Different angles or specific positions
   - Details not visible in video

4. **Combined Assessment**:
   - Key issues across ALL media
   - Priority corrections (most important first)
   - Safety concerns from any angle

Be specific, actionable, and comprehensive."""

    elif videos:
        # Videos only
        if len(videos) == 1:
            prompt = f"""Analyze this exercise video ({len(all_images)} frames):

1. **Exercise**: What is being performed?
2. **Form Rating**: Rate technique 1-10
3. **Consistency**: Does form stay consistent across all reps?
4. **Breakdown Point**: Which rep does technique start to deteriorate?
5. **Key Issues**: Top 2-3 problems
6. **Corrections**: Specific, actionable fixes
7. **Safety**: Any injury risks?
8. **Tone**: Keep it lighthearted but honest. Conversational and encouraging.

Be concise and practical."""
        else:
            prompt = f"""Analyze these {len(videos)} exercise videos:

1. **Exercises**: What is shown in each video?
2. **Form Quality**: Rate each (1-10)
3. **Comparison**: Differences between the videos
4. **Common Issues**: Problems across multiple videos
5. **Priority Corrections**: Most important fixes
6. **Safety**: Any concerns?

Be specific and actionable."""

    else:
        # Images only
        if len(images) == 1:
            prompt = """Analyze this exercise photo:

1. **Exercise**: What is being performed?
2. **Position**: What phase/position of the movement?
3. **Form Rating**: Rate technique 1-10
4. **Key Issues**: Main problems (if any)
5. **Corrections**: Specific fixes needed
6. **Safety**: Any injury risks?

Be specific and actionable."""
        else:
            prompt = f"""Analyze these {len(images)} exercise photos:

1. **Exercise**: What is shown?
2. **What Each Shows**: Different angles or positions?
3. **Form Assessment**: Overall rating 1-10
4. **Issues**: Problems visible from these views
5. **Corrections**: Priority fixes
6. **Safety**: Any concerns?
7. **Tone**: Keep it lighthearted but honest.

Compare and synthesize from all angles."""
    
    # Build messages
    content = []
    for img in all_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {"role": "system", "content": "You are an expert AI spotter analyzing exercise form from photos and videos."},
        {"role": "user", "content": content}
    ]
    
    # Estimate time
    if len(all_images) <= 3:
        time_est = "5-15 seconds"
    elif len(all_images) <= 8:
        time_est = "15-30 seconds"
    elif len(all_images) <= 15:
        time_est = "30-60 seconds"
    else:
        time_est = "60-120 seconds"
    
    print(f"\n⏳ Analyzing {len(all_images)} images (estimated: {time_est})...\n")
    
    # Get analysis
    max_tokens = 600 if len(all_images) <= 5 else 1000
    result = chat_vision(messages, max_new_tokens=max_tokens)
    
    # Display results
    print("="*60)
    print("🔍 ANALYSIS RESULTS")
    print("="*60)
    print(result)
    print("="*60)
    
    print(f"\n✅ Analysis complete!")
    if videos:
        print(f"   📹 Analyzed {len(videos)} video(s) ({sum(8 for _ in videos)} frames)")
    if images:
        print(f"   📸 Analyzed {len(images)} photo(s)")
    
except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install opencv-python transformers torch Pillow")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    print("\nFull error:")
    traceback.print_exc()
