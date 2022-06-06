import pandas as pd
from pycaret.classification import predict_model, load_model
model = load_model('knn')
def load_data(filepath):
    df=pd.read_csv(filepath,index_col='customerID')
    return df

def make_predictions(df);
predictions=predict_model(model, data=df)
predictions.rename({'Label': 'Churn_prediction'}, axis=1,
                   inplace=True)
predictions['Churn_prediction'].replace({1: 'Churn',0:
    'No churn'}, inplace=True)
return predictions['Churn_prediction']

        
if __name__ == "__main__":
    df = load__data('data/new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
    