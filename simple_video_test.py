#!/usr/bin/env python3
"""
Simple video test script
Usage: python simple_video_test.py your_video.mp4
"""

import sys
import os

if len(sys.argv) < 2:
    print("Usage: python simple_video_test.py <video_path>")
    print("\nExample:")
    print("  python simple_video_test.py squat.mp4")
    print("  python simple_video_test.py ~/Desktop/workout.mov")
    sys.exit(1)

VIDEO_PATH = sys.argv[1]

# Check video exists
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Video not found: {VIDEO_PATH}")
    print(f"\nCurrent directory: {os.getcwd()}")
    print("\nMake sure:")
    print("  1. File path is correct")
    print("  2. File extension is included (.mp4, .mov, etc)")
    sys.exit(1)

print("="*60)
print("🎬 GymBot Video Analysis")
print("="*60)
print(f"\nVideo: {VIDEO_PATH}")
print("Processing: Extracting frames...")

try:
    from Spotter_AI import chat_video, build_video_messages
    
    # Build analysis request
    messages = build_video_messages(
        system="You are an expert AI spotter analyzing exercise form. Be specific and actionable.",
        user_text="""Analyze this exercise video:
        
1. **Exercise Name**: What exercise is being performed?
2. **Form Rating**: Rate technique 1-10
3. **Key Issues**: List 2-3 main problems (if any)
4. **Corrections**: Specific fixes needed
5. **Safety**: Any injury risks?

Be concise and practical.""",
        videos=[VIDEO_PATH],
        max_frames_per_video=8  # Good balance of speed/quality
    )
    
    print("\n⏳ Analyzing (this takes 15-40 seconds)...\n")
    
    # Get analysis
    result = chat_video(messages, max_new_tokens=600)
    
    # Display results
    print("="*60)
    print("🔍 AI ANALYSIS")
    print("="*60)
    print(result)
    print("="*60)
    print("\n✅ Analysis complete!")
    
except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install opencv-python transformers torch")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure you're using a VL model:")
    print("     export SPOTTER_MODEL='Qwen/Qwen2-VL-2B-Instruct'")
    print("  2. Check video file is valid (not corrupted)")
    print("  3. Try a different video format (.mp4 works best)")
    
    import traceback
    print("\nFull error:")
    traceback.print_exc()
