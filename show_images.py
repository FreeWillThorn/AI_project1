import numpy as np
import matplotlib.pyplot as plt


def read_dataset(filename):
    with open(filename, 'r') as file:
        number_of_patterns = int(file.readline().strip())
        number_of_pixels = int(file.readline().strip())
        patterns = []
        classes = []
        for line in file:
            data = line.strip().split()
            pixel_data = list(map(float, data[:number_of_pixels]))  # Convert pixel values to floats
            class_data = int(data[number_of_pixels + 3])  # Class label remains an integer
            patterns.append(pixel_data)
            classes.append(class_data)
    return np.array(patterns), np.array(classes)


def display_images(images, title, create_files=False, filename_prefix=''):
    plt.figure(figsize=(10, 10))
    plt.imshow(images, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()
    if create_files:
        plt.savefig(f'{filename_prefix}.png')


def main():
    filenames = ['x24x24.txt', 'y24x24.txt', 'z24x24.txt']
    patterns = []
    classes = []

    # Read all files
    for filename in filenames:
        p, c = read_dataset(filename)
        patterns.append(p)
        classes.append(c)

    patterns = np.vstack(patterns)
    classes = np.concatenate(classes)
    number_of_classes = len(np.unique(classes))
    size_of_pattern = int(np.sqrt(patterns.shape[1]))

    # Process and display mean images
    number_of_classes_in_row = 8
    number_of_rows = (number_of_classes + number_of_classes_in_row - 1) // number_of_classes_in_row
    big_picture = np.zeros((size_of_pattern * number_of_rows, size_of_pattern * number_of_classes_in_row))

    for i in range(number_of_classes):
        class_mask = classes == i
        class_images = patterns[class_mask].reshape(-1, size_of_pattern, size_of_pattern)
        mean_image = np.mean(class_images, axis=0)
        row = i // number_of_classes_in_row
        col = i % number_of_classes_in_row
        big_picture[row * size_of_pattern:(row + 1) * size_of_pattern, col * size_of_pattern:(col + 1) * size_of_pattern] = mean_image

    display_images(big_picture, 'Mean Images',filename_prefix='mean_pattern')

    # Process and display individual examples
    min_num_pat_per_class = min([np.sum(classes == i) for i in range(number_of_classes)])
    for example_index in range(min_num_pat_per_class):
        big_picture = np.zeros((size_of_pattern * number_of_rows, size_of_pattern * number_of_classes_in_row))
        for i in range(number_of_classes):
            class_mask = classes == i
            class_images = patterns[class_mask].reshape(-1, size_of_pattern, size_of_pattern)
            if len(class_images) > example_index:  # Check if the index exists in the class
                image = class_images[example_index]
                row = i // number_of_classes_in_row
                col = i % number_of_classes_in_row
                big_picture[row * size_of_pattern:(row + 1) * size_of_pattern, col * size_of_pattern:(col + 1) * size_of_pattern] = image
        display_images(big_picture, f'Example Images No {example_index + 1}', filename_prefix=f'pattern_{example_index + 1}')

if __name__ == '__main__':
    main()
