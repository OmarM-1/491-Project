#!/usr/bin/env python3
"""
Simple image analysis for exercise form
Analyze one or multiple photos
"""

import sys
import os

print("="*60)
print("📸 GymBot Image Analysis")
print("="*60)

# Get inputs
if len(sys.argv) < 2:
    print("\nUsage:")
    print("  python image_test.py <image1> [image2] [image3] ...")
    print("\nExamples:")
    print("  python image_test.py squat_form.jpg")
    print("  python image_test.py front.jpg side.jpg back.jpg")
    sys.exit(1)

image_paths = sys.argv[1:]

# Verify files exist
for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

print(f"\n📸 Analyzing {len(image_paths)} image(s):")
for i, img in enumerate(image_paths, 1):
    print(f"   {i}. {img}")

print("\n" + "-"*60)

try:
    from Spotter_AI import chat_vision, load_image
    
    # Load all images
    print("📸 Loading images...")
    images = []
    for img_path in image_paths:
        img = load_image(img_path)
        images.append(img)
        print(f"   ✅ Loaded {os.path.basename(img_path)} ({img.size[0]}x{img.size[1]})")
    
    # Build analysis request
    if len(images) == 1:
        prompt = """Analyze this exercise photo:

1. **Exercise Identification**: What exercise is being performed?
2. **Body Position**: Describe the position/phase of the movement
3. **Form Rating**: Rate technique 1-10
4. **Key Issues**: What are the main problems (if any)?
5. **Corrections**: Specific fixes needed
6. **Safety Concerns**: Any injury risks?

Be specific and actionable."""
    else:
        prompt = f"""Analyze these {len(images)} exercise photos (different angles or positions):

1. **Exercise Identification**: What exercise is shown?
2. **What Each Photo Shows**: Briefly describe each angle/position
3. **Form Assessment**: Overall rating 1-10
4. **Issues Across All Angles**: Problems visible from these views
5. **Priority Corrections**: Most important fixes
6. **Safety**: Any concerns from any angle?

Compare and synthesize insights from all angles."""
    
    # Build messages with all images
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {"role": "system", "content": "You are an expert AI spotter analyzing exercise form from photos."},
        {"role": "user", "content": content}
    ]
    
    print("\n⏳ Analyzing (5-15 seconds per image)...\n")
    
    # Get analysis
    result = chat_vision(messages, max_new_tokens=800)
    
    # Display results
    print("="*60)
    print("🔍 FORM ANALYSIS")
    print("="*60)
    print(result)
    print("="*60)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Analyzed {len(images)} image(s)")
    
except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install transformers torch Pillow")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    print("\nFull error:")
    traceback.print_exc()
