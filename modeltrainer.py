
class ModelTrainer:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.models = {
            'SVR_model': SVR(),
            'RF_model': RandomForestRegressor(random_state=123),
            'DT_model': DecisionTreeRegressor(random_state=0),
            'KNN_model': KNeighborsRegressor(n_neighbors=2),
            'LARS': linear_model.Lars(n_nonzero_coefs=1, normalize=False)
        }
        self.results = {}

    def _train_single_model(self, name, model):
        model.fit(self.X_train, self.Y_train)
        Y_pred = model.predict(self.X_test)

        metrics_dict = {
            "Train_Score": model.score(self.X_train, self.Y_train),
            "R_squared": metrics.r2_score(self.Y_test, Y_pred),
            "MAE": metrics.mean_absolute_error(self.Y_test, Y_pred),
            "RMSE": np.sqrt(metrics.mean_squared_error(self.Y_test, Y_pred)),
            "Predictions": Y_pred
        }

        joblib.dump(model, f"saved_models/{name}.pkl")
        return name, metrics_dict

    def train_all_models(self, parallel=True):
        if parallel:
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                results = executor.map(self._train_single_model, self.models.keys(), self.models.values())
        else:
            results = map(lambda kv: self._train_single_model(*kv), self.models.items())

        self.results = dict(results)

    def get_results_dataframe(self):
        df = pd.DataFrame([
            {"Name": name, **metrics} for name, metrics in self.results.items()
        ])
        return df.sort_values("RMSE")

    def visualize_results(self):
        df = self.get_results_dataframe()
        df.plot(x="Name", y=["R_squared", "MAE", "RMSE"], kind="bar", figsize=(10, 5))
        plt.title("Model Performance Comparison")
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.legend(["R-squared", "MAE", "RMSE"])
        plt.grid()
        plt.show()