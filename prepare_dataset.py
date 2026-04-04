import os
import shutil
import csv
import xml.etree.ElementTree as ET

def prepare_kaggle_dataset(raw_data_dir="raw_kaggle_data", 
                           output_image_dir="input_images", 
                           output_csv_path="ground_truth/ground_truth.csv"):
    """
    Crawls the raw Kaggle dataset, pairs XML files with their images, 
    extracts the actual plate number, renames the image, and builds a Ground Truth CSV.
    """
    
    # Ensure output directories exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Valid image extensions to look for
    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
    
    # List to hold our CSV rows
    csv_data = [["Filename", "Actual_Plate"]]
    
    image_counter = 1
    
    print(f"Scanning '{raw_data_dir}' for XML annotations...")

    # Recursively walk through the raw dataset folders (State-wise_OLX, google_images, etc.)
    for root_dir, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root_dir, file)
                base_name = os.path.splitext(file)[0]
                
                # 1. Find the corresponding image file in the same folder
                image_path = None
                image_ext = ""
                for ext in valid_extensions:
                    temp_path = os.path.join(root_dir, base_name + ext)
                    if os.path.exists(temp_path):
                        image_path = temp_path
                        image_ext = ext
                        break
                
                if not image_path:
                    print(f"Warning: Found XML '{file}' but no matching image. Skipping.")
                    continue
                
                # 2. Parse the XML to get the license plate text
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    # PASCAL VOC format usually puts the label inside <object> -> <name>
                    plate_text = ""
                    for obj in root.findall('object'):
                        name_tag = obj.find('name')
                        if name_tag is not None and name_tag.text:
                            plate_text = name_tag.text.strip().upper()
                            break # Assume one primary plate per image
                            
                    if not plate_text:
                        # Fallback if the tag is slightly different
                        name_tag = root.find('.//name')
                        if name_tag is not None and name_tag.text:
                            plate_text = name_tag.text.strip().upper()
                            
                except Exception as e:
                    print(f"Error parsing {file}: {e}")
                    continue
                
                # 3. Rename and copy the image to our clean input_images folder
                new_filename = f"vehicle_{image_counter:04d}{image_ext}"
                new_image_path = os.path.join(output_image_dir, new_filename)
                
                shutil.copy2(image_path, new_image_path)
                
                # 4. Add the pairing to our CSV data
                csv_data.append([new_filename, plate_text])
                print(f"Processed: {base_name} -> {new_filename} | Plate: {plate_text}")
                
                image_counter += 1

    # 5. Write everything to the Ground Truth CSV
    print(f"\nWriting {len(csv_data)-1} records to {output_csv_path}...")
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
        
    print("\nDataset preparation complete! Everything is perfectly synchronized.")

if __name__ == "__main__":
    # Assuming the downloaded dataset is inside a folder named 'raw_kaggle_data'
    # Change the string below if your folder has a different name
    prepare_kaggle_dataset(raw_data_dir="raw_kaggle_data")