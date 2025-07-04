#!/usr/bin/env python3
"""Test different rendering backends to see what works in this environment."""

import os
import sys

def test_osmesa():
    """Test OSMesa software rendering"""
    print("Testing OSMesa...")
    try:
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        import pyrender
        scene = pyrender.Scene()
        r = pyrender.OffscreenRenderer(100, 100)
        print("✓ OSMesa works!")
        return True
    except Exception as e:
        print(f"✗ OSMesa failed: {e}")
        return False

def test_egl():
    """Test EGL rendering"""
    print("\nTesting EGL...")
    try:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        import pyrender
        scene = pyrender.Scene()
        r = pyrender.OffscreenRenderer(100, 100)
        print("✓ EGL works!")
        return True
    except Exception as e:
        print(f"✗ EGL failed: {e}")
        return False

def test_default():
    """Test default rendering"""
    print("\nTesting default (no platform specified)...")
    try:
        if 'PYOPENGL_PLATFORM' in os.environ:
            del os.environ['PYOPENGL_PLATFORM']
        import pyrender
        scene = pyrender.Scene()
        r = pyrender.OffscreenRenderer(100, 100)
        print("✓ Default works!")
        return True
    except Exception as e:
        print(f"✗ Default failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing rendering backends for pyrender...\n")
    
    results = {
        'OSMesa': test_osmesa(),
        'EGL': test_egl(),
        'Default': test_default()
    }
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    
    working = [k for k, v in results.items() if v]
    if working:
        print(f"Working backends: {', '.join(working)}")
        print(f"\nRecommendation: Use {working[0]} by setting:")
        if working[0] != 'Default':
            print(f"export PYOPENGL_PLATFORM={working[0].lower()}")
    else:
        print("No rendering backends are working!")
        print("\nFor training, you likely don't need rendering.")
        print("The dataset loading and data processing should work fine.")