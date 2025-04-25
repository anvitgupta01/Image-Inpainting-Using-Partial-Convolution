import os
import shutil
import argparse
import random
from PIL import Image
from glob import glob
from tqdm import tqdm

def rename_data_folder():
    if os.path.exists("skyview-an-aerial-landscape-dataset/Aerial_Landscapes"):
        shutil.rmtree("data", ignore_errors=True)
        shutil.move("skyview-an-aerial-landscape-dataset/Aerial_Landscapes", "data")
        shutil.rmtree("skyview-an-aerial-landscape-dataset")
        print("✅ Dataset moved and cleaned up.\n")
    else:
        print("❌ Dataset folder not found!\n")

def split_and_rename_dataset(min_size=256):
    output_train = 'data/data_256'
    output_val = 'data/val_256'

    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_val, exist_ok=True)

    class_folders = [f for f in os.listdir("data") if os.path.isdir(os.path.join("data", f))]

    c = 0
    for class_name in tqdm(class_folders, desc="📂 Splitting & renaming"):
        class_path = os.path.join("data", class_name)
        images = glob(os.path.join(class_path, "*.jpg"))
        random.shuffle(images)
        split_idx = int(len(images) * 0.9)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for img in train_imgs:
            dst = os.path.join(output_train, f"{class_name}_{c}.jpg")
            shutil.copy(img, dst)
            c += 1
        for img in val_imgs:
            dst = os.path.join(output_val, f"{class_name}_{c}.jpg")
            shutil.copy(img, dst)
            c += 1

    print("✅ Dataset split complete.\n")

def crop_and_clean(folder_path, min_dim=256, crop_size=(256, 256)):
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist!\n")
        return

    total_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('jpg', 'jpeg', 'png')):
            try:
                img = Image.open(file_path)
                width, height = img.size
                if height < min_dim or width < min_dim:
                    os.remove(file_path)
                else:
                    left = (width - crop_size[0]) / 2
                    top = (height - crop_size[1]) / 2
                    right = (width + crop_size[0]) / 2
                    bottom = (height + crop_size[1]) / 2
                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_img.save(file_path)
                    total_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}\n")
    print(f"✅ {total_count} images retained in {folder_path}\n")

def generate_masks(num_train, num_val):
    os.system("rm -rf data/mask/ data/val_mask/")
    os.system(f"python src/generate_mask.py --dir_name mask --num_mask {num_train}")
    os.system(f"python src/generate_mask.py --dir_name val_mask --num_mask {num_val}")
    print("✅ Masks generated.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation script")
    parser.add_argument('--train_crop_size', type=int, default=256, help='Crop size for training images')
    parser.add_argument('--val_crop_size', type=int, default=256, help='Crop size for validation images')
    args = parser.parse_args()

    rename_data_folder()
    split_and_rename_dataset()
    crop_and_clean("data/data_256", crop_size=(args.train_crop_size, args.train_crop_size))
    crop_and_clean("data/val_256", crop_size=(args.val_crop_size, args.val_crop_size))

    num_train = len(os.listdir("data/data_256"))
    num_val = len(os.listdir("data/val_256"))

    generate_masks(num_train, num_val)
