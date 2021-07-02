DEBUG = True

if DEBUG:
    from PIL import Image
    import numpy as np
 
    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)

DATA_DIR = 'C:/Users/shubh/Documents/projects/ocr/data/'
TEST_DIR = 'test/'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels.idx1-ubyte'
TEST_DATA_FILENAME = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 't10k-labels.idx1-ubyte'


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')



def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)		#magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row= []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)	
                image.append(row)
            images.append(image)

    return images




def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)		#magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]
    
    
    
def extract_features(x):
    return [flatten_list(sample) for sample in x]

def dist(x,y):
    return sum(
        [
              (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2 
              for x_i,y_i in zip(x,y)
        ]
    )**0.5



def get_training_distances_for_test_sample(x_train,test_sample):
    return [dist(train_sample, test_sample) for train_sample in x_train]



def get_most_frequent_element(l):
    return max(l , key=l.count)



def knn(x_train, y_train, x_test, k):
    y_pred = []
    for test_sample_idx,test_sample in enumerate(x_test):
        training_distances = get_training_distances_for_test_sample(x_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x:x[1]
            )
        ]
        candidates = [
            bytes_to_int(y_train[idx])	
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    return y_pred




def main():
    X_train = read_images(TRAIN_DATA_FILENAME,20000)
    y_train = read_labels(TRAIN_LABELS_FILENAME,20000)
    X_test = read_images(TEST_DATA_FILENAME,300)
    y_test = read_labels(TEST_LABELS_FILENAME,300)
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))

    if DEBUG:
        for idx,test_sample in enumerate(X_test):
            write_image(test_sample, f'{TEST_DIR}{idx}.png')	
        X_test = [read_image(f'{DATA_DIR}our_test.png')]
    
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)
 
    y_pred = knn(X_train, y_train, X_test, 3)
    print(y_pred)
    accuracy = sum([
        int(y_pred_i == y_test_i)
        for y_pred_i , y_test_i
        in zip(y_test, y_pred)
    ]) / len(y_test)
    print(accuracy)


if __name__ == '__main__':
    main()