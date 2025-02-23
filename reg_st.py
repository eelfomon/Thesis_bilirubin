
def extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(1, (50, 50))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(dst, cv2.COLOR_BGR2Lab)
    ycbcr = cv2.cvtColor(dst, cv2.COLOR_BGR2YCrCb)

    features = {
        "R": np.mean(dst[:, :, 2]), "G": np.mean(dst[:, :, 1]), "B": np.mean(dst[:, :, 0]),
        "H": np.mean(hsv[:, :, 0]), "S": np.mean(hsv[:, :, 1]), "V": np.mean(hsv[:, :, 2]),
        "L": np.mean(lab[:, :, 0]), "A": np.mean(lab[:, :, 1]), "BB": np.mean(lab[:, :, 2]),
        "Y": np.mean(ycbcr[:, :, 0]), "Cb": np.mean(ycbcr[:, :, 1]), "Cr": np.mean(ycbcr[:, :, 2])
    }
    return features

image_paths = glob.glob("all_images/*.jpg")
data = [extract_features(img) for img in image_paths]
df = pd.DataFrame(data)

df['TSB'] = np.random.uniform(1, 10, len(df))

X = df.drop(columns=['TSB'])
Y = df['TSB']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

models = {
    'SVR': SVR(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=3),
    'LARS': linear_model.Lars(n_nonzero_coefs=1, normalize=False)
}

results = []
for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    r2 = r2_score(Y_test, Y_pred)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)

    results.append({'Model': name, 'R2 Score': r2, 'RMSE': rmse})

results_df = pd.DataFrame(results).sort_values(by="RMSE")
print(results_df)

plt.figure(figsize=(10,6))
X_ax = range(len(Y_train))
plt.plot(X_ax, Y_train, label='Actual', color='orange', linestyle='-', marker='o')
plt.plot(X_ax, models['SVR'].predict(X_train), label='SVR Prediction', color='blue', linestyle='-', marker='o')
plt.xlabel('Training Sample')
plt.ylabel('Bilirubin Level (TSB)')
plt.legend()
plt.title(' Output')
plt.show()