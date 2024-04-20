from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.utils import shuffle
import xgboost as xgb

# Preprocess
def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'
    
def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
    
    if ((fecha >= range1_min and fecha <= range1_max) or 
        (fecha >= range2_min and fecha <= range2_max) or 
        (fecha >= range3_min and fecha <= range3_max) or
        (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0
    
def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff

def shuffle_data(data):
    training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM','delay']], random_state = 111)
    return training_data

def get_dummies_features(data):
    features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1
    )

    return features

def get_top_10_features(features):
    top_10_features = ["OPERA_Latin American Wings", "MES_7","MES_10","OPERA_Grupo LATAM",
                       "MES_12", "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"]
    
    top_10_features = features[top_10_features]

    return top_10_features

# Fit model
def split_data(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        target, test_size = 0.33,
                                                        random_state = 42)

    return x_train, x_test, y_train, y_test

def get_scale(y_train):
    n_y0 = len(y_train[y_train == 0])
    n_y1 = len(y_train[y_train == 1])
    scale = n_y0/n_y1

    return scale

def train_model(x_train, y_train, scale):
    xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
    xgb_model.fit(x_train, y_train)
    
    return xgb_model

# Get predictions
def get_prediction(model_path: str,
                   features: pd.DataFrame):
    
    trained_model = XGBClassifier()
    trained_model.load_model(model_path)
    
    # Realizar predicciones con el modelo
    model_predictions = trained_model.predict(features)
    
    return model_predictions.tolist()