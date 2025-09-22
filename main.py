
from src.utils.input import CSVLoader, YAMLLoader
from src.transform import DataTransform
from src.model import DataModelling
import mlflow
import datetime


if __name__ == "__main__" :
        
    config = YAMLLoader().load_file("params.yaml")
    input_path = config["data"]["raw_data"]

    df = CSVLoader().load_file(input_path)
    transform = DataTransform(df,config)
    df_feature = transform.select_features()
    df_transform = transform.transform_date()
    df_groupby = transform.group_by()
    modelling = DataModelling(df_groupby,config)
    series = modelling.df_timeseries()
    X_train,X_test = modelling.split_timeseries(series)
    results = modelling.train_model(X_train,X_test)
    
    mlflow.set_experiment("Baseline_Models")

    for result in results:
        model_name=result["Model"]
        trial_name = f"Model_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=trial_name):
            mlflow.log_metric("SMAPE", result["SMAPE"])
            mlflow.log_metric("MSE", result["MSE"])
            mlflow.log_param("model_type", "baseline")
            mlflow.set_tag("Model",model_name)
            mlflow.set_tag("Author","AA")

    #print("Model_Results_Evaluate:")
    #for result in results:
        #print(result)


    

    





        


 