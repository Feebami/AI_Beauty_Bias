import os
import shutil

os.makedirs('MEB_cropped_images', exist_ok=True)
for gender in os.listdir('MEB_images'):
    for race in os.listdir(os.path.join('MEB_images', gender)):
        for file in os.listdir(os.path.join('MEB_images', gender, race)):
            source_path = os.path.join('MEB_images', gender, race, file)
            destination_path = os.path.join('MEB_cropped_images', file)

            shutil.copy(source_path, destination_path)