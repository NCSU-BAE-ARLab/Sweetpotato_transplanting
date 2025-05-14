# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "overlap"

# set export dir
export_dir = "overlap/coco"

# set train split rate
train_split_rate = 0.9

# set category ID start value
category_id_start = 1

# convert labelme annotations to coco
# use_customize_function is True for my special case only, where I modified the labelme2coco package to suit my needs
labelme2coco.convert(labelme_folder, export_dir, train_split_rate, category_id_start=category_id_start,use_customize_function=True)
