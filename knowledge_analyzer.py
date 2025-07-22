#!/usr/bin/env python3
"""
Single Model Complete Advertisement Analyzer - FIXED VERSION
Uses config/prompts.py - ONE model does EVERYTHING
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from PIL import Image
import time
import logging
from datetime import datetime
import base64
from typing import List, Dict, Any, Optional
import re

# Import YOUR prompt configuration
from config.prompts import (
    get_universal_prompt,
    validate_json_structure,
    extract_confidence_scores,
    extract_selling_points,
    extract_visual_elements,
    extract_persuasion_techniques,
    CONFIDENCE_THRESHOLDS
)

# Structured output libraries
import instructor
from pydantic import BaseModel, Field

# Provider imports
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import torch
    from transformers import (
        LlavaNextProcessor, LlavaNextForConditionalGeneration,
        LlavaForConditionalGeneration, LlavaProcessor,
        AutoProcessor, AutoModelForCausalLM
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Simple .env loader
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

load_env_file()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def encode_image_base64(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def fix_json_string(json_str):
    """Fix common JSON formatting issues"""
    try:
        # Remove markdown code blocks
        json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'^```\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'\s*```$', '', json_str, flags=re.MULTILINE)
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Add missing commas before closing braces (simple fix)
        json_str = re.sub(r'(["\d\]}])\s*\n\s*(["}])', r'\1,\n  \2', json_str)
        
        return json_str.strip()
    except:
        return json_str

def extract_json_from_text(text):
    """Extract JSON from text with fixing attempts"""
    # Try to find JSON patterns
    json_patterns = [
        r'\{.*\}',  # Basic JSON object
        r'```json\s*(\{.*?\})\s*```',  # Markdown JSON block
        r'```\s*(\{.*?\})\s*```',  # Generic code block
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Try original first
            try:
                json.loads(match)
                return match
            except:
                # Try to fix it
                try:
                    fixed_json = fix_json_string(match)
                    json.loads(fixed_json)
                    return fixed_json
                except:
                    continue
    
    # Try to fix the entire text
    try:
        fixed_text = fix_json_string(text)
        json.loads(fixed_text)
        return fixed_text
    except:
        pass
    
    return text

# Simplified Analysis Result Class
class SimpleAnalysisResult:
    """Simple container for analysis results"""
    def __init__(self, raw_output="", confidence="low", completeness=0.1):
        self.raw_output = raw_output
        self.confidence = confidence  
        self.completeness = completeness
        self.parsed_json = None
        self.validation_result = None
    
    def model_dump(self):
        """Convert to dict for storage"""
        try:
            if self.parsed_json:
                return self.parsed_json
            else:
                # Create a basic structure from raw output
                return {
                    "visual_elements": {
                        "objects": [{"name": "analysis_provided", "prominence": 0.5, "description": self.raw_output[:200]}],
                        "people": [],
                        "text": [],
                        "colors": [],
                        "composition": {"layout": "unknown", "focal_point": "center", "visual_flow": "unclear", "style": "unknown"}
                    },
                    "brand_marketing": {
                        "primary_brand": {"name": "see_raw_analysis", "confidence": 0.4, "visual_elements": []},
                        "product_or_service": {"name": "check_raw_output", "category": "unknown", "features_highlighted": []},
                        "secondary_brands": [],
                        "marketing_message": {"primary_message": self.raw_output[:100], "supporting_messages": [], "tone": "unknown"}
                    },
                    "selling_points": [{"selling_point": "analysis_available", "confidence": 0.4, "evidence": [], "explicitness": "implicit"}],
                    "target_audience": {
                        "primary_demographic": {"age_range": "unknown", "gender": "unknown", "income_level": "unknown", "lifestyle": "unknown"},
                        "psychographics": [],
                        "values_targeted": [],
                        "emotional_appeals": []
                    },
                    "persuasion_strategy": {
                        "techniques": [],
                        "visual_metaphors": [],
                        "urgency_indicators": [],
                        "trust_signals": []
                    },
                    "cultural_context": {
                        "setting": {"location_type": "unknown", "geographic_indicators": [], "time_period": "unknown"},
                        "cultural_references": [],
                        "values_conveyed": [],
                        "stereotypes_present": [],
                        "inclusivity_indicators": []
                    },
                    "effectiveness_analysis": {
                        "strengths": [{"strength": "model_provided_analysis", "impact": "check_raw_output"}],
                        "potential_weaknesses": [{"weakness": "json_parsing_issues", "impact": "reduced_structure"}],
                        "clarity_score": 4,
                        "visual_impact_score": 4,
                        "message_coherence_score": 4,
                        "overall_effectiveness_score": 4
                    },
                    "extraction_metadata": {
                        "completeness_self_assessment": self.completeness,
                        "confidence_level": self.confidence,
                        "challenging_aspects": ["json_formatting"],
                        "assumptions_made": ["simplified_structure"],
                        "extraction_quality_indicators": {
                            "text_readability": "medium",
                            "image_quality": "unknown",
                            "cultural_context_clarity": "medium"
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Error in model_dump: {e}")
            return {"error": "model_dump_failed", "raw_output": str(self.raw_output)[:500]}

# Local Model Handler - SIMPLIFIED
class LocalModelAnalyzer:
    """Simplified local model handler"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the local model"""
        if not HF_AVAILABLE:
            raise ImportError("transformers/torch not installed")
        
        logger.info(f"🔄 Loading LOCAL model: {self.model_name}")
        
        try:
            if "llava-v1.6" in self.model_name or "llava-next" in self.model_name:
                self.processor = LlavaNextProcessor.from_pretrained(self.model_name, use_fast=True)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            elif "llava" in self.model_name:
                self.processor = LlavaProcessor.from_pretrained(self.model_name, use_fast=True)
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            logger.info("✅ LOCAL model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            raise
    
    def analyze_image_complete(self, image_path):
        """Analyze image with retry logic"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Get YOUR universal prompt
            universal_prompt = get_universal_prompt()
            
            # Enhanced prompt for better JSON
            enhanced_prompt = f"""{universal_prompt}

CRITICAL: Respond ONLY with valid JSON. No explanations.
- Use double quotes for all strings
- Add commas between elements
- Close all braces properly

JSON Response:"""

            # Try 2 attempts with different settings
            for attempt in range(2):
                temp = 0.1 if attempt == 0 else 0.2
                logger.info(f"Attempt {attempt + 1}/2 (temp={temp})")
                
                try:
                    if "llava" in self.model_name:
                        full_prompt = f"USER: <image>\n{enhanced_prompt}\nASSISTANT:"
                        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
                        
                        for key in inputs:
                            if torch.is_tensor(inputs[key]):
                                inputs[key] = inputs[key].to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=2000,
                                do_sample=True,
                                temperature=temp,
                                pad_token_id=self.processor.tokenizer.eos_token_id
                            )
                        
                        response = self.processor.decode(outputs[0], skip_special_tokens=True)
                        
                        if "ASSISTANT:" in response:
                            raw_output = response.split("ASSISTANT:")[-1].strip()
                        else:
                            raw_output = response.strip()
                    
                    else:
                        # Generic approach
                        inputs = self.processor(images=image, text=enhanced_prompt, return_tensors="pt")
                        
                        for key in inputs:
                            if torch.is_tensor(inputs[key]):
                                inputs[key] = inputs[key].to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model.generate(**inputs, max_new_tokens=2000, temperature=temp)
                        
                        raw_output = self.processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Try to parse JSON
                    json_str = extract_json_from_text(raw_output)
                    
                    try:
                        parsed_json = json.loads(json_str)
                        validation_result = validate_json_structure(parsed_json)
                        
                        if validation_result["valid"] or validation_result["completeness_score"] > 0.5:
                            logger.info(f"✅ Success on attempt {attempt + 1} (completeness: {validation_result['completeness_score']:.2f})")
                            
                            result = SimpleAnalysisResult(
                                raw_output=raw_output,
                                confidence="high" if validation_result["completeness_score"] > 0.8 else "medium",
                                completeness=validation_result["completeness_score"]
                            )
                            result.parsed_json = parsed_json
                            result.validation_result = validation_result
                            return result
                        else:
                            logger.warning(f"Attempt {attempt + 1} low completeness: {validation_result['completeness_score']:.2f}")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Attempt {attempt + 1} JSON error: {e}")
                        if attempt == 1:  # Last attempt
                            return SimpleAnalysisResult(
                                raw_output=raw_output,
                                confidence="low",
                                completeness=0.3
                            )
                
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == 1:
                        return SimpleAnalysisResult(
                            raw_output=f"Generation failed: {str(e)}",
                            confidence="low",
                            completeness=0.1
                        )
            
            # Fallback
            return SimpleAnalysisResult(
                raw_output="All attempts failed",
                confidence="low", 
                completeness=0.1
            )
            
        except Exception as e:
            logger.error(f"Error in analyze_image_complete: {e}")
            return SimpleAnalysisResult(
                raw_output=f"Error: {str(e)}",
                confidence="low",
                completeness=0.1
            )

# API Model Handler - SIMPLIFIED
class APIModelAnalyzer:
    """Simplified API model handler"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.instructor_client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup API client"""
        if "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            
            client = openai.OpenAI(api_key=api_key)
            self.instructor_client = instructor.from_openai(client)
            logger.info("✅ OpenAI client configured")
            
        elif "gemini" in self.model_name.lower():
            if not GEMINI_AVAILABLE:
                raise ImportError("google-genai not installed")
            
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")
            
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.instructor_client = instructor.from_gemini(client)
            logger.info("✅ Gemini client configured")
        else:
            raise ValueError(f"Unsupported API model: {self.model_name}")
    
    def analyze_image_complete(self, image_path):
        """Analyze with API using instructor"""
        try:
            # Get YOUR universal prompt
            universal_prompt = get_universal_prompt()
            
            if "gpt" in self.model_name.lower():
                base64_image = encode_image_base64(image_path)
                
                # Simple text response from API
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": universal_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=3000
                )
                
                raw_output = response.choices[0].message.content
                
            else:  # Gemini
                import google.generativeai as genai
                uploaded_file = genai.upload_file(image_path)
                
                response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content([
                    universal_prompt,
                    uploaded_file
                ])
                
                raw_output = response.text
            
            # Parse the response
            try:
                json_str = extract_json_from_text(raw_output)
                parsed_json = json.loads(json_str)
                validation_result = validate_json_structure(parsed_json)
                
                result = SimpleAnalysisResult(
                    raw_output=raw_output,
                    confidence="high" if validation_result["completeness_score"] > 0.8 else "medium",
                    completeness=validation_result["completeness_score"]
                )
                result.parsed_json = parsed_json
                result.validation_result = validation_result
                return result
                
            except Exception as e:
                logger.warning(f"API JSON parsing failed: {e}")
                return SimpleAnalysisResult(
                    raw_output=raw_output,
                    confidence="medium",
                    completeness=0.6
                )
                
        except Exception as e:
            logger.error(f"API analysis failed: {e}")
            return SimpleAnalysisResult(
                raw_output=f"API Error: {str(e)}",
                confidence="low",
                completeness=0.1
            )

# Main Analyzer - SIMPLIFIED
class ConfigPromptAnalyzer:
    """Simplified analyzer using YOUR config/prompts.py system"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_calls_count = 0
        
        # Determine model type and setup
        if any(indicator in model_name.lower() for indicator in ['llava', 'hf']):
            self.model_type = "local"
            self.analyzer = LocalModelAnalyzer(model_name)
            logger.info(f"🏠 Using LOCAL model: {model_name}")
        else:
            self.model_type = "api"
            self.analyzer = APIModelAnalyzer(model_name)
            logger.info(f"☁️ Using API model: {model_name}")
    
    def analyze_image(self, image_path):
        """Analyze image using YOUR prompt system"""
        image_id = Path(image_path).stem
        image_name = Path(image_path).name
        
        logger.info(f"🔍 Analyzing: {image_name}")
        
        try:
            # Analyze with chosen model
            analysis_result = self.analyzer.analyze_image_complete(image_path)
            
            if self.model_type == "api":
                self.api_calls_count += 1
            
            # Get analysis data
            analysis_data = analysis_result.model_dump()
            
            # Extract elements using YOUR functions if possible
            extracted_elements = {}
            try:
                if analysis_result.parsed_json:
                    extracted_elements = {
                        'confidence_scores': extract_confidence_scores(analysis_result.parsed_json),
                        'selling_points': extract_selling_points(analysis_result.parsed_json),
                        'visual_elements': extract_visual_elements(analysis_result.parsed_json),
                        'persuasion_techniques': extract_persuasion_techniques(analysis_result.parsed_json)
                    }
            except Exception as e:
                logger.warning(f"Element extraction failed: {e}")
                extracted_elements = {'extraction_failed': str(e)}
            
            # Create result structure
            result = {
                'image_path': str(image_path),
                'image_name': image_name,
                'image_id': image_id,
                'model_used': self.model_name,
                'model_type': self.model_type,
                'analysis_data': analysis_data,
                'extracted_elements': extracted_elements,
                'raw_output': analysis_result.raw_output,
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_metadata': {
                    'file_size_mb': round(os.path.getsize(image_path) / (1024 * 1024), 2),
                    'image_dimensions': self._get_image_dimensions(image_path),
                    'config_prompts_used': True,
                    'api_calls_made': 1 if self.model_type == "api" else 0,
                    'confidence_level': analysis_result.confidence,
                    'completeness_score': analysis_result.completeness
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'image_name': image_name,
                'image_id': image_id,
                'model_used': self.model_name,
                'model_type': self.model_type,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _get_image_dimensions(self, image_path):
        """Get image dimensions"""
        try:
            with Image.open(image_path) as img:
                return {"width": img.width, "height": img.height}
        except:
            return {"width": 0, "height": 0}

# Utility Functions
def create_output_structure(model_name):
    """Create output structure"""
    clean_model_name = model_name.lower().replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")
    
    print(f"📁 Creating output structure:")
    print(f"   📂 Model folder: {clean_model_name}")
    
    output_dir = Path("output") / "fixed_analysis" / clean_model_name
    
    if output_dir.exists():
        print(f"   🔄 Clearing existing folder: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Created: {output_dir}")
    
    return {
        'output_dir': output_dir,
        'model_name': clean_model_name,
        'base_output': Path("output")
    }

def save_analysis_results(results, output_structure):
    """Save analysis results"""
    output_dir = output_structure['output_dir']
    saved_files = []
    successful_analyses = []
    failed_analyses = []
    
    for result in results:
        image_id = result['image_id']
        filename = f"{image_id}.json"
        file_path = output_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            saved_files.append(file_path)
            
            if 'error' not in result:
                successful_analyses.append(result)
                print(f"   💾 Saved: {filename}")
            else:
                failed_analyses.append(result)
                print(f"   ❌ Saved with error: {filename}")
                
        except Exception as e:
            print(f"   ❌ Failed to save {filename}: {e}")
            failed_analyses.append(result)
    
    return {
        'saved_files': saved_files,
        'successful_analyses': successful_analyses,
        'failed_analyses': failed_analyses,
        'success_rate': len(successful_analyses) / len(results) if results else 0
    }

def generate_summary(save_result, output_structure):
    """Generate summary"""
    successful = save_result['successful_analyses']
    
    if not successful:
        print("❌ No successful analyses to summarize")
        return
    
    completeness_scores = []
    confidence_levels = []
    model_types = {}
    
    for result in successful:
        model_type = result.get('model_type', 'unknown')
        model_types[model_type] = model_types.get(model_type, 0) + 1
        
        completeness = result.get('processing_metadata', {}).get('completeness_score', 0.0)
        completeness_scores.append(completeness)
        
        confidence = result.get('processing_metadata', {}).get('confidence_level', 'unknown')
        confidence_levels.append(confidence)
    
    summary = {
        'fixed_analysis_summary': {
            'total_images': len(successful) + len(save_result['failed_analyses']),
            'successful_analyses': len(successful),
            'success_rate': save_result['success_rate'],
            'model_used': output_structure['model_name'],
            'model_types_used': model_types,
            'average_completeness': sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0,
            'confidence_distribution': {level: confidence_levels.count(level) for level in set(confidence_levels)},
            'analysis_timestamp': datetime.now().isoformat()
        }
    }
    
    # Save summary
    summary_path = output_structure['output_dir'] / f"fixed_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Fixed Analysis Summary:")
    print(f"   ✅ Success rate: {save_result['success_rate']:.1%}")
    print(f"   🔧 Model types: {model_types}")
    print(f"   📊 Avg completeness: {summary['fixed_analysis_summary']['average_completeness']:.2f}")
    print(f"   🎯 Confidence distribution: {summary['fixed_analysis_summary']['confidence_distribution']}")
    print(f"   💾 Summary saved: {summary_path}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='FIXED Advertisement Analyzer using config/prompts.py')
    parser.add_argument('--num_images', '-n', type=int, default=10,
                       help='Number of images to process')
    parser.add_argument('--dataset_dir', '-d', type=str,
                       default="pitt_ads/cars/images",
                       help='Path to dataset directory')
    parser.add_argument('--model', '-m', type=str, default='gemini-2.0-flash-001',
                       help='Model to use')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Show available providers
    print(f"🔧 Available providers:")
    print(f"   Gemini: {'✅' if GEMINI_AVAILABLE else '❌'}")
    print(f"   OpenAI: {'✅' if OPENAI_AVAILABLE else '❌'}")
    print(f"   HuggingFace: {'✅' if HF_AVAILABLE else '❌'}")
    
    if HF_AVAILABLE and torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize analyzer
    try:
        analyzer = ConfigPromptAnalyzer(args.model)
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Create output structure
    output_structure = create_output_structure(args.model)
    
    # Find images
    ads_dir = Path(args.dataset_dir)
    if not ads_dir.exists():
        print(f"❌ Directory not found: {ads_dir}")
        return
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    all_images = []
    for ext in image_extensions:
        all_images.extend(ads_dir.rglob(f"*{ext}"))
        all_images.extend(ads_dir.rglob(f"*{ext.upper()}"))
    
    selected_images = all_images[:args.num_images]
    
    print(f"\n🎯 FIXED Advertisement Analyzer")
    print(f"🤖 Model: {args.model}")
    print(f"🔧 Model type: {analyzer.model_type}")
    print(f"📁 Dataset: {ads_dir}")
    print(f"🖼️ Images: {len(selected_images)}")
    print(f"🧠 Pipeline: YOUR config/prompts.py system (FIXED)")
    print(f"💾 Output: {output_structure['output_dir']}")
    print("=" * 70)
    
    all_results = []
    start_time = datetime.now()
    
    for i, img_path in enumerate(selected_images, 1):
        print(f"\n[{i}/{len(selected_images)}] 🔍 Processing: {img_path.name}")
        
        result = analyzer.analyze_image(img_path)
        all_results.append(result)
        
        if 'error' not in result:
            confidence = result.get('processing_metadata', {}).get('confidence_level', 'unknown')
            completeness = result.get('processing_metadata', {}).get('completeness_score', 0.0)
            model_type = result.get('model_type', 'unknown')
            print(f"   ✅ Analysis complete ({model_type}, confidence: {confidence}, completeness: {completeness:.2f})")
        else:
            print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        print("-" * 50)
        time.sleep(1)
    
    # Save results
    print(f"\n💾 Saving analysis results...")
    save_result = save_analysis_results(all_results, output_structure)
    
    # Generate summary
    summary = generate_summary(save_result, output_structure)
    
    # Final summary
    duration = datetime.now() - start_time
    print(f"\n🎉 FIXED Analysis Complete!")
    print(f"⏱️ Duration: {duration}")
    print(f"🖼️ Images processed: {len(selected_images)}")
    print(f"✅ Success rate: {save_result['success_rate']:.1%}")
    print(f"🔧 API calls: {analyzer.api_calls_count}")
    print(f"📁 Results saved to: {output_structure['output_dir']}")

if __name__ == "__main__":
    main()