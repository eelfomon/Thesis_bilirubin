
def load_data():
    def read_json_data(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def process_images(image_paths):
        processed_images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (480, 480))
            processed_images.append(img)
        return np.array(processed_images) / 255

    def extract_info(data):
        image_paths = [item['image_path'] for item in data]
        tsb_values = [item['TSB'] for item in data]
        return image_paths, tsb_values

    json_file_path = '/E/Thesis/Bilirubin_dataset/Data.json'

    data = read_json_data(json_file_path)

    image_paths, tsb_values = extract_info(data)

    images = process_images(image_paths)

    return images, np.array(tsb_values)

def split_data(X, Y, test_size=0.2, shuffle=True, random_state=1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=shuffle, random_state=random_state)
    return X_train, X_test, Y_train, Y_test