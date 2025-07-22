import os
import requests
import json
import time
import random
from pathlib import Path
from urllib.parse import urljoin

def show_available_topics():
    """Display all available topics"""
    
    print("Loading available topics...")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    base_url = "https://people.cs.pitt.edu/~mzhang/visualization/dataset/"
    
    try:
        data_url = urljoin(base_url, "ads_maj_topic_two_sents.json")
        response = requests.get(data_url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        print("\n" + "="*50)
        print("AVAILABLE TOPICS:")
        print("="*50)
        
        topics_list = []
        for i, topic in enumerate(sorted(data.keys()), 1):
            image_count = sum(len(sentiments) for sentiments in data[topic].values())
            print(f"{i:2d}. {topic:<25} ({image_count:4d} images)")
            topics_list.append(topic)
        
        return data, topics_list
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def get_user_input(data, topics_list):
    """Get user input for topic and sentiment"""
    
    print("\n" + "="*50)
    print("SELECT TOPIC:")
    print("="*50)
    
    # Get topic
    while True:
        topic_input = input(f"\nEnter topic name or number (1-{len(topics_list)}): ").strip()
        
        if topic_input.isdigit():
            topic_num = int(topic_input)
            if 1 <= topic_num <= len(topics_list):
                selected_topic = topics_list[topic_num - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(topics_list)}")
        elif topic_input.lower() in [t.lower() for t in topics_list]:
            # Find exact match
            for topic in topics_list:
                if topic.lower() == topic_input.lower():
                    selected_topic = topic
                    break
            break
        else:
            print("Invalid topic. Please enter a valid topic name or number.")
    
    # Show sentiments available for this topic
    print(f"\nSelected topic: {selected_topic}")
    
    if selected_topic in data:
        available_sentiments = sorted(data[selected_topic].keys())
        
        print("\n" + "="*50)
        print(f"AVAILABLE SENTIMENTS FOR {selected_topic.upper()}:")
        print("="*50)
        
        for i, sentiment in enumerate(available_sentiments, 1):
            image_count = len(data[selected_topic][sentiment])
            print(f"{i:2d}. {sentiment:<15} ({image_count:3d} images)")
        
        print("\n(Press Enter to download ALL sentiments for this topic)")
        sentiment_input = input("Enter sentiment name or number (or press Enter for all): ").strip()
        
        selected_sentiment = None
        if sentiment_input:
            if sentiment_input.isdigit():
                sent_num = int(sentiment_input)
                if 1 <= sent_num <= len(available_sentiments):
                    selected_sentiment = available_sentiments[sent_num - 1]
                else:
                    print(f"Invalid number. Using all sentiments.")
            elif sentiment_input.lower() in [s.lower() for s in available_sentiments]:
                for sentiment in available_sentiments:
                    if sentiment.lower() == sentiment_input.lower():
                        selected_sentiment = sentiment
                        break
            else:
                print(f"'{sentiment_input}' not found for {selected_topic}. Using all sentiments.")
    
    return selected_topic, selected_sentiment

def download_images(data, annotations_data, topic, sentiment=None):
    """Download images and Q&A annotations for the selected topic and sentiment"""
    
    # Create folder structure: pitt_ads/topic/ or pitt_ads/topic_sentiment/
    base_folder = "pitt_ads"
    if sentiment:
        topic_folder = f"{topic}_{sentiment}"
        print(f"\nDownloading {topic} images with {sentiment} sentiment...")
    else:
        topic_folder = topic
        print(f"\nDownloading ALL {topic} images...")
    
    # Create folders
    main_folder = os.path.join(base_folder, topic_folder)
    images_folder = os.path.join(main_folder, "images")
    annotations_folder = os.path.join(main_folder, "annotations")
    
    Path(images_folder).mkdir(parents=True, exist_ok=True)
    Path(annotations_folder).mkdir(parents=True, exist_ok=True)
    
    # Set headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Get images for the topic
    all_images = []
    if topic in data:
        if sentiment:
            # Specific sentiment
            if sentiment in data[topic]:
                all_images = data[topic][sentiment]
                print(f"Found {len(all_images)} images for {topic} + {sentiment}")
            else:
                print(f"No images found for {topic} + {sentiment}")
                return
        else:
            # All sentiments
            for sent, images in data[topic].items():
                all_images.extend(images)
                print(f"  Found {len(images)} images for {topic} + {sent}")
    else:
        print(f"Topic '{topic}' not found")
        return
    
    if not all_images:
        print("No images to download")
        return
    
    # Remove duplicates
    all_images = list(set(all_images))
    print(f"\nTotal unique images: {len(all_images)}")
    
    downloaded = 0
    skipped = 0
    
    # Download each image and its annotations
    for i, image_id in enumerate(all_images, 1):
        try:
            print(f"[{i}/{len(all_images)}] Processing {image_id}")
            
            # Get base filename without extension
            base_name = os.path.splitext(image_id)[0]
            
            # Check if already downloaded
            image_path = os.path.join(images_folder, image_id)
            annotation_path = os.path.join(annotations_folder, f"{base_name}.json")
            
            if os.path.exists(image_path) and os.path.exists(annotation_path):
                print(f"  Skipping {image_id} (already exists)")
                skipped += 1
                continue
            
            # Download image
            if not os.path.exists(image_path):
                image_success = download_image_by_id(image_id, images_folder, headers)
                if not image_success:
                    continue
            else:
                image_success = True
            
            # Save Q&A annotations with topic/sentiment info
            if not os.path.exists(annotation_path):
                save_annotations(image_id, annotations_data, data, annotations_folder)
            
            if image_success:
                downloaded += 1
                print(f"  Saved: {image_id} + annotations")
            
            # Be polite
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  Error processing {image_id}: {e}")
    
    print(f"\n" + "="*50)
    print("DOWNLOAD COMPLETE!")
    print("="*50)
    print(f"Downloaded: {downloaded} new images")
    print(f"Skipped: {skipped} existing images")
    print(f"Total images: {downloaded + skipped}")
    print(f"Images folder: {images_folder}/")
    print(f"Annotations folder: {annotations_folder}/")

def download_image_by_id(image_id, images_folder, headers):
    """Download image using the ID-to-URL logic"""
    
    # Replicate the id2url function from JavaScript
    image_num = int(image_id.split(".")[0])
    
    if image_num >= 170000:
        image_url = f"https://people.cs.pitt.edu/~mzhang/image_ads/10/{image_id}"
    else:
        last_digit = str(image_num)[-1]
        image_url = f"https://people.cs.pitt.edu/~mzhang/image_ads/{last_digit}/{image_id}"
    
    try:
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        filepath = os.path.join(images_folder, image_id)
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        return True
        
    except Exception as e:
        print(f"    Error downloading image: {e}")
        return False

def clean_text(text):
    """Clean HTML tags and normalize text"""
    if isinstance(text, str):
        # Replace <br> tags with newlines
        text = text.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
        # Remove other HTML tags if any
        import re
        text = re.sub(r'<[^>]+>', '', text)
        # Clean up extra whitespace
        text = text.strip()
        return text
    return text

def clean_annotation_data(data):
    """Clean HTML tags from annotation data"""
    if isinstance(data, list):
        return [clean_text(item) if isinstance(item, str) else item for item in data]
    elif isinstance(data, str):
        return clean_text(data)
    return data

def save_annotations(image_id, annotations_data, data, annotations_folder):
    """Save Q&A annotations with topic and sentiment tags for an image"""
    
    base_id = os.path.splitext(image_id)[0]  # Remove extension
    
    # Try to find annotations
    annotation = None
    if image_id in annotations_data:
        annotation = annotations_data[image_id]
    elif base_id in annotations_data:
        annotation = annotations_data[base_id]
    
    # Find which topics and sentiments this image belongs to
    image_topics = []
    image_sentiments = []
    
    for topic, topic_data in data.items():
        for sentiment, images in topic_data.items():
            if image_id in images:
                if topic not in image_topics:
                    image_topics.append(topic)
                if sentiment not in image_sentiments:
                    image_sentiments.append(sentiment)
    
    # Structure the annotation data with cleaned text
    if annotation:
        structured_annotation = {
            "image_id": image_id,
            "topics": sorted(image_topics),
            "sentiments": sorted(image_sentiments),
            "strategy": clean_annotation_data(annotation.get("strategy", [])),
            "slogans": clean_annotation_data(annotation.get("slogans", [])), 
            "QA": clean_annotation_data(annotation.get("QA", []))
        }
    else:
        # Create annotation with topic/sentiment info even if no Q&A found
        structured_annotation = {
            "image_id": image_id,
            "topics": sorted(image_topics),
            "sentiments": sorted(image_sentiments),
            "strategy": [],
            "slogans": [],
            "QA": []
        }
    
    # Save as JSON
    filename = f"{base_id}.json"
    filepath = os.path.join(annotations_folder, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(structured_annotation, f, indent=2, ensure_ascii=False)

def main():
    """Main function"""
    
    print("Pitt Image Ads Dataset Downloader")
    print("="*50)
    
    # Show available topics and get data
    data, topics_list = show_available_topics()
    if not data:
        return
    
    # Load annotations
    print("Loading Q&A annotations...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    base_url = "https://people.cs.pitt.edu/~mzhang/visualization/dataset/"
    
    try:
        annot_url = urljoin(base_url, "annotations_maj_topic_two_sents.json") 
        annot_response = requests.get(annot_url, headers=headers, timeout=15)
        annot_response.raise_for_status()
        annotations_data = annot_response.json()
        print(f"Loaded annotations for {len(annotations_data)} images")
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return
    
    # Get user input
    topic, sentiment = get_user_input(data, topics_list)
    
    # Download images and annotations
    download_images(data, annotations_data, topic, sentiment)

if __name__ == "__main__":
    main()