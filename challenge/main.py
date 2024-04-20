import model
import pandas as pd


delay_model = model.DelayModel()
data = pd.read_csv('./data/data.csv')
features, target = delay_model.preprocess(data=data, target_column='delay')
trained_model = delay_model.fit(features, target)

model_path = 'challenge/trained_model/xgboost_model.pkl'
data_to_predict = pd.read_csv('./data/x_test.csv')
predictions = delay_model.predict(model_path=model_path, features=data_to_predict)

print(predictions)