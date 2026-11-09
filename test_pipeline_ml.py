import unittest
from generate_and_train import generate_housing_dataset, preprocess_and_train


class TestPredictionLoyersML(unittest.TestCase):
    def test_dataset_generation(self) -> None:
        df = generate_housing_dataset(n_samples=300)
        self.assertEqual(len(df), 300)
        self.assertIn("loyer_mensuel_fcfa", df.columns)
        self.assertIn("surface_m2", df.columns)
        self.assertTrue((df["loyer_mensuel_fcfa"] > 0).all())

    def test_random_forest_r2_score(self) -> None:
        df = generate_housing_dataset(n_samples=850, random_state=42)
        metrics = preprocess_and_train(df, random_state=42)
        self.assertGreaterEqual(metrics["r2_score"], 0.72)
        self.assertLessEqual(metrics["r2_score"], 0.85)
        self.assertIn("surface_m2", metrics["top_features"])


if __name__ == "__main__":
    unittest.main()
