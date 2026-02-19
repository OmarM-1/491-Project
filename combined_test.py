#!/usr/bin/env python3
"""
Combined video + image analysis
Analyze workout video plus additional form check photos
"""

import sys
import os

print("="*60)
print("🎬📸 GymBot Combined Video + Image Analysis")
print("="*60)

# Get inputs
if len(sys.argv) < 2:
    print("\nUsage:")
    print("  python combined_test.py <video> [image1] [image2] ...")
    print("\nExamples:")
    print("  python combined_test.py squat_set.mov")
    print("  python combined_test.py squat_set.mov side_view.jpg front_view.jpg")
    sys.exit(1)

video_path = sys.argv[1]
image_paths = sys.argv[2:] if len(sys.argv) > 2 else []

# Verify files exist
if not os.path.exists(video_path):
    print(f"❌ Video not found: {video_path}")
    sys.exit(1)

for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

print(f"\n📹 Video: {video_path}")
if image_paths:
    print(f"📸 Additional images: {len(image_paths)}")
    for i, img in enumerate(image_paths, 1):
        print(f"   {i}. {img}")
else:
    print("📸 No additional images")

print("\n" + "-"*60)
print("Processing...")

try:
    from Spotter_AI import chat_vision, load_image, load_video
    
    # Load video frames
    print("🎬 Extracting video frames...")
    video_frames = load_video(video_path, max_frames=8)
    print(f"   ✅ Extracted {len(video_frames)} frames from video")
    
    # Load additional images
    additional_images = []
    if image_paths:
        print("📸 Loading additional images...")
        for img_path in image_paths:
            img = load_image(img_path)
            additional_images.append(img)
            print(f"   ✅ Loaded {os.path.basename(img_path)}")
    
    # Combine all images
    all_images = video_frames + additional_images
    
    # Build comprehensive analysis request
    if additional_images:
        prompt = f"""You are analyzing {len(video_frames)} frames from a workout video, plus {len(additional_images)} additional photo(s).

**Video frames (first {len(video_frames)} images):** Show the complete exercise set

**Additional photos (last {len(additional_images)} image(s)):** Different angles or form checks

**Analysis requested:**
1. **Exercise Identification**: What exercise is being performed?
2. **Video Set Analysis**: 
   - Form quality across the set (1-10)
   - Consistency across reps
   - When does form break down?
3. **Additional Photo Analysis**:
   - What do the extra photos show?
   - Any issues visible from these angles?
4. **Combined Assessment**:
   - Key issues across ALL images
   - Priority corrections
   - Safety concerns

Be specific and actionable."""
    else:
        prompt = f"""Analyze this exercise video ({len(video_frames)} frames):

1. **Exercise Name**: What is this?
2. **Form Rating**: Rate 1-10
3. **Consistency**: Does form stay consistent across all reps?
4. **Breakdown Point**: Which rep does technique deteriorate?
5. **Key Issues**: Top 2-3 problems
6. **Corrections**: Specific fixes needed
7. **Safety**: Any injury risks?
8. **Tone**: Keep it informal and encouraging. Lighthearted but honest.

Be concise and practical."""
    
    # Build messages with all images
    content = []
    for img in all_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {"role": "system", "content": "You are an expert AI spotter analyzing exercise form from multiple angles and perspectives."},
        {"role": "user", "content": content}
    ]
    
    print("\n⏳ Analyzing (this may a couple minutes with multiple images)...\n")
    
    # Get analysis
    result = chat_vision(messages, max_new_tokens=1000)
    
    # Display results
    print("="*60)
    print("🔍 COMPREHENSIVE ANALYSIS")
    print("="*60)
    print(result)
    print("="*60)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Analyzed: {len(video_frames)} video frames + {len(additional_images)} photos")
    
except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install opencv-python transformers torch Pillow")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    print("\nFull error:")
    traceback.print_exc()
