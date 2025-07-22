#!/usr/bin/env python3
"""
Test Gemini Model Image Support
Tests different Gemini models to see which ones support images
"""

import os
import google.generativeai as genai
from PIL import Image
import io
import base64

# Load .env file if it exists
def load_env_file():
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
    except FileNotFoundError:
        pass

# Load environment
load_env_file()

def create_test_image():
    """Create a simple test image"""
    from PIL import Image, ImageDraw
    
    # Create a simple colored rectangle
    img = Image.new('RGB', (200, 100), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "TEST IMAGE", fill='black')
    draw.rectangle([10, 30, 190, 90], outline='red', width=3)
    
    return img

def test_model_with_image(model_name):
    """Test if a specific model supports images"""
    
    print(f"\n🧪 Testing model: {model_name}")
    print("-" * 50)
    
    try:
        # Create model
        model = genai.GenerativeModel(model_name)
        
        # Create test image
        test_image = create_test_image()
        
        # Test with image
        prompt = "Describe what you see in this image."
        
        print(f"📤 Sending image + prompt to {model_name}...")
        response = model.generate_content([prompt, test_image])
        
        print(f"✅ {model_name} SUPPORTS IMAGES!")
        print(f"📝 Response: {response.text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} FAILED!")
        print(f"🚨 Error: {str(e)[:100]}...")
        return False

def list_available_models():
    """List all available Gemini models"""
    print("📋 Available Gemini Models:")
    print("-" * 30)
    
    try:
        models = genai.list_models()
        for model in models:
            if 'gemini' in model.name.lower():
                supports_images = 'generateContent' in model.supported_generation_methods
                image_support = "🖼️ " if supports_images else "📝 "
                print(f"{image_support}{model.name}")
    except Exception as e:
        print(f"❌ Couldn't list models: {e}")

def test_image_support():
    """Test image support for different Gemini models"""
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ No API key found!")
        return False
    
    print(f"🔑 Using API key: {api_key[:10]}...{api_key[-5:]}")
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # List available models first
        list_available_models()
        
        # Test specific models
        models_to_test = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro-vision',
            'gemini-pro'
        ]
        
        print(f"\n🧪 Testing Image Support")
        print("=" * 50)
        
        results = {}
        
        for model_name in models_to_test:
            try:
                results[model_name] = test_model_with_image(model_name)
            except Exception as e:
                print(f"\n❌ {model_name}: Not available - {e}")
                results[model_name] = False
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print("=" * 30)
        
        image_supporting_models = []
        
        for model, supports in results.items():
            status = "✅ Supports Images" if supports else "❌ No Image Support"
            print(f"{model}: {status}")
            if supports:
                image_supporting_models.append(model)
        
        # Recommendation
        print(f"\n🎯 RECOMMENDATION:")
        if image_supporting_models:
            print(f"✅ Use these models for image analysis:")
            for model in image_supporting_models:
                print(f"   - {model}")
                
            # Update script recommendation
            if 'gemini-1.5-flash' in image_supporting_models:
                print(f"\n✅ Your script is correct! gemini-1.5-flash supports images.")
            else:
                print(f"\n⚠️  Update your script to use: {image_supporting_models[0]}")
                print(f"   Change model_name='{image_supporting_models[0]}'")
        else:
            print("❌ No models support images with your current API access")
            print("💡 Try: gemini-pro-vision or check your API permissions")
        
        return len(image_supporting_models) > 0
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    print("🖼️  Gemini Image Support Tester")
    print("=" * 40)
    
    success = test_image_support()
    
    if success:
        print(f"\n🎉 Ready for image analysis!")
    else:
        print(f"\n❌ Fix image support issues first")
        print("💡 Solutions:")
        print("   1. Use gemini-pro-vision for images")
        print("   2. Check API permissions")
        print("   3. Verify billing is enabled")