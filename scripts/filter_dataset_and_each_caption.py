import csv
import shutil
import os 

csv = csv.reader(open('metadata.csv', 'r'))
filtered_rows = [row for row in csv]

print(filtered_rows[1])

# for image_path, caption in filtered_rows[1:]:
#     data_path = 'dataset_images/' + image_path
#     new_data_path = 'datasets/' + image_path

#     with open(f'datasets/{image_path.split(".")[0]}.txt', 'w') as f:
#         f.write(caption)

#     shutil.move(data_path, new_data_path)
    

for image_path, caption in filtered_rows[1:]:
    data_path = 'datasets/' + image_path

    text_path = 'datasets/' + image_path.split(".")[0] + '.txt'

    if (image_path.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']):
        os.remove(data_path)
        os.remove(text_path)
        print(f'Removed {data_path} and {text_path}'
              )