import os


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))

    train_images = os.path.join(root_dir, 'data/enhanced_crowdsat/train/img_aug_rename')
    val_images = os.path.join(root_dir, 'data/enhanced_crowdsat/val/img')

    train_txt_path = os.path.join(root_dir, 'data/crowdsat/crowd_train.list')
    val_txt_path = os.path.join(root_dir, 'data/crowdsat/crowd_val.list')

    os.makedirs(os.path.dirname(train_txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_txt_path), exist_ok=True)

    image_files = os.listdir(train_images)
    image_files.sort()
    list_file = open(train_txt_path, 'w')
    for train_img in image_files:
        list_file.write(f"{os.path.join(train_images, train_img)} {os.path.join(train_images.replace('img', 'txts'), train_img.replace('.png', '.txt'))}\n")

    image_files = os.listdir(val_images)
    image_files.sort()
    list_file = open(val_txt_path, 'w')
    for train_img in image_files:
        list_file.write(f"{os.path.join(val_images, train_img)} {os.path.join(val_images.replace('img', 'txts'), train_img.replace('.png', '.txt'))}\n")

if __name__ == '__main__':
    main()
