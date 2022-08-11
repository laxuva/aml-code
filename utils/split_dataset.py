from data.segmentation_dataset import SegmentationDataset


def split_dataset(image_path, label_path):
    dataset_train, dataset_val, dataset_test = SegmentationDataset.load_train_val_and_test_data(
        image_path, label_path, preload_percentage=0
    )

    dataset_train.save("train_dataset.json")
    dataset_val.save("val_dataset.json")
    dataset_test.save("test_dataset.json")


if __name__ == '__main__':
    split_dataset("~\\Documents\\data\\aml\\masked128png", "~\\Documents\\data\\aml\\seg_mask128png")
