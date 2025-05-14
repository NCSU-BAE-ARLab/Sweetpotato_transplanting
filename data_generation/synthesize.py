
import os
import json
import random
import csv
from PIL import Image, ImageEnhance
from pil_augment import augment_image
from tqdm import tqdm

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def adjust_coordinates(points, dx, dy, img_width, img_height, canvas_center_x, canvas_center_y):
    adjusted_points = []
    for point in points:
        x = canvas_center_x + (point[0] - img_width // 2) + dx
        y = canvas_center_y + (point[1] - img_height // 2) + dy

        x = clamp(x, 0, 2*canvas_center_x - 1)
        y = clamp(y, 0, 2*canvas_center_y - 1)
        adjusted_points.append([x, y])
    return adjusted_points

def calculate_boundaries(points, canvas_width, canvas_height, img_width, img_height, canvas_center_x, canvas_center_y):
    adjusted_points = adjust_coordinates(points, 0, 0, img_width, img_height, canvas_center_x, canvas_center_y)
    min_x = min(adjusted_points, key=lambda x: x[0])[0]
    max_x = max(adjusted_points, key=lambda x: x[0])[0]
    min_y = min(adjusted_points, key=lambda x: x[1])[1]
    max_y = max(adjusted_points, key=lambda x: x[1])[1]

    max_dx_positive = int(canvas_width - max_x)
    max_dx_negative = int(-min_x)
    max_dy_positive = int(canvas_height - max_y)
    max_dy_negative = int(-min_y)

    return max_dx_negative, max_dx_positive, max_dy_negative, max_dy_positive


folder1_path = 'singles'
folder2_path = 'singles' 
output_folder_png = 'short_slip_dataset_new'
output_folder_json = 'short_slip_dataset_new'

os.makedirs(output_folder_png, exist_ok=True)
os.makedirs(output_folder_json, exist_ok=True)

images = [f for f in os.listdir(folder1_path) if f.endswith('.png')]
random.shuffle(images)  # random
canvas_width = 640 #960
canvas_height = 480 #960
canvas_center_x = canvas_width // 2
canvas_center_y = canvas_height // 2

records = []

for i in tqdm(range(0, 4500)):
    selected_images = random.sample(images, random.randint(1, 30))
    composite_image = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    all_shapes = []

    background = Image.open('background_color.jpg')

    #Introducing some randomness for data diversity
    background =  augment_image(background).convert("RGBA")



    composite_image.paste(background, (0,0), background)

    for j,image_name in enumerate(selected_images):
        img = Image.open(os.path.join(folder1_path, image_name))
        img_width, img_height = img.size

        #Introducing some randomness for data diversity
        img =  augment_image(img,object_instance=True)

        json_path = os.path.join(folder2_path, image_name.replace('.png', '.json'))
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

            # for shape in data['shapes']:
            shape_mask = data['shapes'][0]
            points = shape_mask['points']
            max_dx_negative, max_dx_positive, max_dy_negative, max_dy_positive = calculate_boundaries(
                points, canvas_width, canvas_height, img_width, img_height, canvas_center_x, canvas_center_y
            )
            dx = random.randint(max_dx_negative, max_dx_positive)
            dy = random.randint(max_dy_negative, max_dy_positive)

            records.append([i, dx, dy])

            # processing all shapes
            for shape in data['shapes']:
                adjusted_points = adjust_coordinates(shape['points'], dx, dy, img_width, img_height, canvas_center_x, canvas_center_y)
                new_shape = {
                    "label": shape["label"],
                    "points": adjusted_points,
                    "group_id": j,
                    "shape_type": shape.get("shape_type", "point"),
                    "flags": shape.get("flags", {}),
                    "mask": shape.get("mask", None)
                }
                all_shapes.append(new_shape)


        paste_x = canvas_center_x + dx - img_width // 2
        paste_y = canvas_center_y + dy - img_height // 2

        # img = img.convert("RGBA")
        composite_image.paste(img, (paste_x, paste_y), img)
        # composite_image = composite_image.convert("RGB")

    composite_image_filename = f'image_{i}.png'
    composite_json_filename = f'image_{i}.json'
    composite_image_path = os.path.join(output_folder_png, composite_image_filename)
    composite_json_path = os.path.join(output_folder_json, composite_json_filename)
    composite_image.save(composite_image_path)

    composite_json = {
        "version": "5.4.1",
        "flags": {},
        "shapes": all_shapes,
        "imagePath": composite_image_filename,
        "imageData": None,
        "imageHeight": canvas_height,
        "imageWidth": canvas_width
    }
    with open(composite_json_path, 'w') as json_file:
        json.dump(composite_json, json_file, indent=4)

csv_file_path = os.path.join(output_folder_json, "short-multi-noise-20-24_records.csv")
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["iteration", "dx", "dy"])
    writer.writerows(records)
 



