import pandas as pd
from pycaret.classification import predict_model, load_model
model = load_model('knn')
def load_data(filepath):
    df=pd.read_csv(filepath,index_col='Patient number')
    return df

def make_predictions(df);
predictions=predict_model(model, data=df)
predictions.rename({'Label': 'Diabetes_prediction'}, axis=1,
                   inplace=True)
predictions['Diabetes_prediction'].replace({1: 'Diabetes',0:
    'No diabetes'}, inplace=True)
return predictions['Diabetes_prediction']

        
if __name__ == "__main__":
    df = load__data('data/new_diabetes_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
    