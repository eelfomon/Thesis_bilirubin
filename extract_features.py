
def extract_features(image_path):
    img = cv2.imread(image_path)

    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    labImg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    ycbcrImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    features = {
        'R': np.mean(img[:, :, 2]), 'G': np.mean(img[:, :, 1]), 'B': np.mean(img[:, :, 0]),
        'H': np.mean(hsvImg[:, :, 0]), 'S': np.mean(hsvImg[:, :, 1]), 'V': np.mean(hsvImg[:, :, 2]),
        'L': np.mean(labImg[:, :, 0]), 'A': np.mean(labImg[:, :, 1]), 'BB': np.mean(labImg[:, :, 2]),
        'Y': np.mean(ycbcrImg[:, :, 0]), 'Cr': np.mean(ycbcrImg[:, :, 1]), 'Cb': np.mean(ycbcrImg[:, :, 2])
    }
    return features

def process_images(image_paths):
    features_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_features, image_paths))
        features_list.extend(results)

    return pd.DataFrame(features_list)