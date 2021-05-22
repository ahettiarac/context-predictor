from PredictorRNN import PredictorRNN
from similarity_predictor import SimilarityPredictor


similarity = SimilarityPredictor()
result = similarity.prepare_for_prediction("Hello World","Hi World")
print("Cosine Similarity Score: {}".format(result))

