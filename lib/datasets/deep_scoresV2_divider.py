from sklearn.model_selection import train_test_split
import json
import os
import os



def reduce_train_dataset(reduce_size=0.5):
    dataset_prefix = '../../data/DeepScores_2020/'

    train_json_path = dataset_prefix + 'deepscores_train.json'

    with open(train_json_path, 'r') as f:
        json_data = json.load(f)

    images = json_data['images']
    annotations = json_data['annotations']

    train_image_ids, val_image_ids, _, _ = train_test_split(
        [image['id'] for image in images],
        [0] * len(images),
        test_size=reduce_size,
        random_state=42
    )

    train_images = [image for image in images if image['id'] in train_image_ids]

    train_annotations = {}
    for idx, ann in annotations.items():
        if int(ann['img_id']) in train_image_ids:
            train_annotations[idx] = ann

    train_data = {
        "info": json_data['info'],
        "images": train_images,
        "annotation_sets": json_data['annotation_sets'],
        "annotations": train_annotations,
        "categories": json_data['categories']
    }

    with open(dataset_prefix + 'deepscores_train_slim.json', 'w') as f:
        json.dump(train_data, f)

    print('done')


def divide_train_validation_set(test_size=0.2):
    dataset_prefix = '../../data/DeepScores_2020/'

    train_json_path = dataset_prefix + 'deepscores_train.json'

    with open(train_json_path, 'r') as f:
        json_data = json.load(f)

    images = json_data['images']
    annotations = json_data['annotations']

    train_image_ids, val_image_ids, _, _ = train_test_split(
        [image['id'] for image in images],
        [0] * len(images),
        test_size=test_size,
        random_state=42
    )

    val_images = [image for image in images if image['id'] in val_image_ids]

    val_annotations = {}
    for idx, ann in annotations.items():
        if int(ann['img_id']) in val_image_ids:
            val_annotations[idx] = ann


    val_data = {
        "info": json_data['info'],
        "images": val_images,
        "annotation_sets": json_data['annotation_sets'],
        "annotations": val_annotations,
        "categories": json_data['categories']
    }

    print(f'train image count : {len(images)}')
    print(f'train data count : {len(annotations)}')
    print(f'validation image count : {len(val_images)}')
    print(f'validation data count : {len(val_data["annotations"])}')

    with open(dataset_prefix + 'deepscores_val.json', 'w') as f:
        json.dump(val_data, f)

    print("done!")

def read_dataset_for_debug():
    dataset_prefix = '../../data/DeepScores_2020/'

    json_path = dataset_prefix + 'deepscores_val.json'

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    images = json_data['images']
    annotations = json_data['annotations']

    print(len(images))
    print(len(annotations))

if __name__ == '__main__':
    # read_dataset_for_debug()
    divide_train_validation_set()
    # reduce_train_dataset(0.5)
