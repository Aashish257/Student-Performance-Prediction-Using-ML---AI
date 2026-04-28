import joblib
import numpy as np
import pandas as pd
import os

class StudentModel:
    def __init__(self, model_path='model.sav'):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'sex', 'age', 'address', 'Medu', 'Fedu', 
            'traveltime', 'failures', 'paid', 'higher', 
            'internet', 'goout', 'G1', 'G2'
        ]
        self._load_model()

    def _load_model(self):
        """Loads the serialized model from disk."""
        if os.path.exists(self.model_path):
            try:
                # Using joblib as it's more efficient for scikit-learn models
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                # Fallback to pickle if joblib fails (though joblib is usually better)
                import pickle
                self.model = pickle.load(open(self.model_path, 'rb'))
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def predict(self, data_dict):
        """
        Predicts student performance based on input data.
        data_dict should contain keys matching self.feature_names
        """
        if self.model is None:
            self._load_model()

        # Extract features in correct order
        try:
            processed_features = []
            for f in self.feature_names:
                val = data_dict.get(f, 0)
                # Handle empty strings from form inputs
                if val == '' or val is None:
                    processed_features.append(0.0)
                else:
                    processed_features.append(float(val))
            
            X = np.array(processed_features).reshape(1, -1)
            df = pd.DataFrame(X, columns=self.feature_names)
            
            prediction = self.model.predict(df)
            return prediction[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def get_feature_importance(self):
        """Returns feature importance if the model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            import numpy as np
            import pandas as pd
            # Convert numpy floats to standard Python floats for JSON serialization
            importance_values = [float(val) for val in self.model.feature_importances_]
            return dict(zip(self.feature_names, importance_values))
        return None
