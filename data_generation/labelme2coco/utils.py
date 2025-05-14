import json
import os

import jsonschema

image_schema = {
    "type": "object",
    "properties": {"file_name": {"type": "string"}, "id": {"type": "integer"}},
    "required": ["file_name", "id"],
}

segmentation_schema = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {
            "type": "number",
        },
        "additionalItems": False,
    },
    "additionalItems": False,
}

annotation_schema = {
    "type": "object",
    "properties": {
        "image_id": {"type": "integer"},
        "category_id": {"type": "integer"},
        "segmentation": segmentation_schema,
    },
    "required": ["image_id", "category_id", "segmentation"],
}

category_schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
    "required": ["name", "id"],
}

coco_schema = {
    "type": "object",
    "properties": {
        "images": {"type": "array", "items": image_schema, "additionalItems": False},
        "annotations": {
            "type": "array",
            "items": annotation_schema,
            "additionalItems": False,
        },
        "categories": {
            "type": "array",
            "items": category_schema,
            "additionalItems": False,
        },
    },
    "required": ["images", "annotations", "categories"],
}


def read_and_validate_coco_annotation(coco_annotation_path: str) -> (dict, bool):
    """
    Reads coco formatted annotation file and validates its fields.
    """
    try:
        with open(coco_annotation_path) as json_file:
            coco_dict = json.load(json_file)
        jsonschema.validate(coco_dict, coco_schema)
        response = True
    except jsonschema.exceptions.ValidationError as e:
        print("well-formed but invalid JSON:", e)
        response = False
    except json.decoder.JSONDecodeError as e:
        print("poorly-formed text, not JSON:", e)
        response = False

    return coco_dict, response

def update_visibility_for_keypoints(annotations,bmask):
    for j,ann in enumerate(annotations):
        try:
            keypoints = ann.keypoints
            total_keypoints = int(len(keypoints)/3)
            for i in range(total_keypoints):
                x,y,v = int(keypoints[3*i]),int(keypoints[3*i+1]),keypoints[3*i+2]
                if v == 2:
                    if bmask[y, x] == 255:
                        keypoints[3*i+2] = 1 # point is hidden
            ann.keypoints = keypoints
            annotations[j] = ann
            # print('keypoints found:',ann.category_name)
            # print(keypoints)
        except:
            # print('no keypoints found:',ann.category_name)
            continue
    return annotations