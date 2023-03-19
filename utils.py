import pands as pd

from datasets import ImageDataset


def to_dataframe():
    dataset = ImageDataset(data_dir)
    df = pd.DataFrame({
        "image_path": dataset.image_paths,
        "label": dataset.labels
    })
    return df


def plot_examples(images):
    fig = plt.figure(figsize=(15, 15))
    columns = 6
    rows = 5

    for i in range(1, len(images)):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()