""""
Random Forest Regressor

Author: Selene Ledain
Feb. 6th 2023
"""

from sklearn.ensemble import RandomForestRegressor
from utils.utils_pixel import *
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

# MLFlow
import mlflow
import mlflow.sklearn
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



n_timesteps_in = 9
n_timesteps_out = 1
n_feats_in = 85
n_feats_out = 2
exp = 'nofilter'
sample_type = 'pixel_data'
feature_set = ["MSI", "NDMI", "NDVIre", "NDWI", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "CP", "NDVI", "SCL",
               "VPDX", "VPDN", "AWVC", "MAXT", "MINT", "SR" ,"SP", "T", "TCC", "TP", 
               "DEM", 
               "CURV17", "CURV19", "CURV21", "EAST17", "EAST19", "EAST21", "NORT17", "NORT19", "NORT21", "RUGG17", "RUGG19", "RUGG21", "SLOP17", "SLOP19", "SLOP21", "TWI17", "TWI19", "TWI21",
               "CLAY0_5", "CLAY100_200", "CLAY15_30", "CLAY30_60", "CLAY5_15", "CLAY60_100",
               "FED0_5", "FED100_200", "FED15_30",  "FED30_60", "FED5_15", "FED60_100",
               "FC", "FH",
               "GRAV0_5", "GRAV100_200", "GRAV15_30", "GRAV30_60", "GRAV5_15", "GRAV60_100",
               "SAND0_5", "SAND100_200", "SAND15_30", "SAND30_60","SAND5_15", "SAND60_100",
               "SDEP",
               "CARB0_5", "CARB100_200", "CARB15_30", "CARB30_60", "CARB5_15", "CARB60_100",
               "PH0_5", "PH100_200", "PH15_30", "PH30_60","PH5_15", "PH60_100"]
remove_band = ["MSI", "NDMI", "NDVIre", "NDWI", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "SCL",
               "VPDX", "VPDN", "AWVC", "MAXT", "MINT", "SP", "TCC",
               "CURV17", "CURV19", "CURV21", "EAST17", "EAST19", "EAST21", "NORT17", "NORT19", "NORT21", "RUGG17", "RUGG19", "RUGG21", "SLOP17", "SLOP19", "SLOP21", "TWI17", "TWI19", "TWI21",
               "CLAY0_5", "CLAY100_200", "CLAY15_30", "CLAY30_60", "CLAY5_15", "CLAY60_100",
               "FED0_5", "FED100_200", "FED15_30",  "FED30_60", "FED5_15", "FED60_100",
               "FC", "FH",
               "GRAV0_5", "GRAV100_200", "GRAV15_30", "GRAV30_60", "GRAV5_15", "GRAV60_100",
               "SAND0_5", "SAND100_200", "SAND15_30", "SAND30_60","SAND5_15", "SAND60_100",
               "SDEP",
               "CARB0_5", "CARB100_200", "CARB15_30", "CARB30_60", "CARB5_15", "CARB60_100",
               "PH0_5", "PH100_200", "PH15_30", "PH30_60","PH5_15", "PH60_100"]


n_epochs = 100
n_batches = 1000
n_batches_val = 100
n_batches_te = 100
batch_size = 40


# Create new MLflow experiment
now = datetime.now()
dt_string = 'RF_basic'+now.strftime("%d/%m/%Y")
#dt_string = 'debug'
mlflow.create_experiment(name=dt_string) 
experiment = mlflow.get_experiment_by_name(dt_string)


with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=f'val'):
    mlflow.log_param(f"n_batches", n_batches_val)
    mlflow.log_param("batch_size", batch_size)
    
# Get run that was just created and its ID to use when tracking
client = mlflow.tracking.MlflowClient() # Create a MlflowClient object
runs = client.search_runs(experiment.experiment_id)
mlflow_run_id_val = [r.info.run_id for r in runs if r.info.run_name==f'val'][0]



def get_val_metrics(rf, batch_size, n_batches_val, sample_type, exp, n_timesteps_in, n_timesteps_out, feature_set, remove_band):

    r2_tot = 0
    mse_tot = 0
    
    for batch_nbr in range(n_batches_val):
        # get the current batch of data
        X_tr = np.empty([batch_size, n_timesteps_in, len(feature_set)-len(remove_band)])
        y_tr = np.empty([batch_size, n_timesteps_out, 2])

        # Load a batch here 
        batch_nbr=50
        img, label = load_batch(batch_size=batch_size, batch_nbr=batch_nbr, sample_type=sample_type, split='val', exp=exp, n_timesteps_out=n_timesteps_out)
        # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]

        img = remove_features_batch_level(img, remove_band, feature_set)
        label = remove_features_batch_level(label, remove_band, feature_set)

        # still predict both, but dont use cp_idx for masking clouds in loss
        to_keep = [0, 1]
        label = label[:,:,to_keep,:,:]

        X_tr[:, :, :] = th.reshape(img, (batch_size, img.size(dim=1), img.size(dim=2))).numpy()
        y_tr[:, :, :] = th.reshape(label, (batch_size, label.size(dim=1), label.size(dim=2))).numpy()

        # Concatenate all timesteps after each other
        X_batch = np.reshape(X_tr, (batch_size, n_timesteps_in*(len(feature_set)-len(remove_band))))
        y_batch = np.reshape(y_tr, (batch_size, n_timesteps_out*n_feats_out))
        
        # calculate the MSE loss
        y_pred = rf.predict(X_batch)
        mse_tot += mean_squared_error(y_batch, y_pred)
        r2_tot += r2_score(y_batch, y_pred)

    return r2_tot/n_batches_val, mse_tot/n_batches_val


def get_test_metrics(rf, batch_size, n_batches_te, sample_type, exp, n_timesteps_in, n_timesteps_out, feature_set, remove_band):
    
    r2_tot = 0
    mse_tot = 0
    
    for batch_nbr in range(n_batches_te):
        # get the current batch of data
        X_tr = np.empty([batch_size, n_timesteps_in, len(feature_set)-len(remove_band)])
        y_tr = np.empty([batch_size, n_timesteps_out, 2])

        # Load a batch here 
        batch_nbr=50
        img, label = load_batch(batch_size=batch_size, batch_nbr=batch_nbr, sample_type=sample_type, split='test', exp=exp, n_timesteps_out=n_timesteps_out)
        # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]

        img = remove_features_batch_level(img, remove_band, feature_set)
        label = remove_features_batch_level(label, remove_band, feature_set)

        # still predict both, but dont use cp_idx for masking clouds in loss
        to_keep = [0, 1]
        label = label[:,:,to_keep,:,:]

        X_tr[:, :, :] = th.reshape(img, (batch_size, img.size(dim=1), img.size(dim=2))).numpy()
        y_tr[:, :, :] = th.reshape(label, (batch_size, label.size(dim=1), label.size(dim=2))).numpy()

        # Concatenate all timesteps after each other
        X_batch = np.reshape(X_tr, (batch_size, n_timesteps_in*(len(feature_set)-len(remove_band))))
        y_batch = np.reshape(y_tr, (batch_size, n_timesteps_out*n_feats_out))

        # calculate the MSE loss
        y_pred = rf.predict(X_batch)
        mse_tot += mean_squared_error(y_batch, y_pred)
        r2_tot += r2_score(y_batch, y_pred)
    
    return r2_tot/n_batches_te, mse_tot/n_batches_te





rf = RandomForestRegressor()


for epoch in tqdm(range(n_epochs)):
    for batch_nbr in range(n_batches):
        # get the current batch of data
        X_tr = np.empty([batch_size, n_timesteps_in, len(feature_set)-len(remove_band)])
        y_tr = np.empty([batch_size, n_timesteps_out, 2])

        # Load a batch here 
        img, label = load_batch(batch_size=batch_size, batch_nbr=batch_nbr, sample_type=sample_type, split='train', exp=exp, n_timesteps_out=n_timesteps_out)
        # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]

        img = remove_features_batch_level(img, remove_band, feature_set)
        label = remove_features_batch_level(label, remove_band, feature_set)

        # still predict both, but dont use cp_idx for masking clouds in loss
        to_keep = [0, 1]
        label = label[:,:,to_keep,:,:]

        X_tr[:, :, :] = th.reshape(img, (batch_size, img.size(dim=1), img.size(dim=2))).numpy()
        y_tr[:, :, :] = th.reshape(label, (batch_size, label.size(dim=1), label.size(dim=2))).numpy()

        # Concatenate all timesteps after each other
        X_batch = np.reshape(X_tr, (batch_size, n_timesteps_in*(len(feature_set)-len(remove_band))))
        y_batch = np.reshape(y_tr, (batch_size, n_timesteps_out*n_feats_out))

        
        # fit the regressor on the current batch
        rf.fit(X_batch, y_batch)
        
        r2, mse = get_val_metrics(rf, batch_size, n_batches_val, sample_type, exp, n_timesteps_in, n_timesteps_out, feature_set, remove_band)
        print(f'Validation metrics for epoch {epoch} batch {batch_nbr}: R2 {r2}. MSE {mse}.') 
        client.log_metric(mlflow_run_id_val,"r2 val in train", r2, step=epoch*n_batches + batch_nbr)
        client.log_metric(mlflow_run_id_val,"mse val in train", mse, step=epoch*n_batches + batch_nbr)
    
    
    r2, mse = get_val_metrics(rf, batch_size, n_batches_val, sample_type, exp, n_timesteps_in, n_timesteps_out, feature_set, remove_band)
    print(f'Validation metrics for epoch {epoch}: R2 {r2}. MSE {mse}.') 
    client.log_metric(mlflow_run_id_val,"r2", r2, step=epoch)
    client.log_metric(mlflow_run_id_val,"mse", mse, step=epoch)
    
    


# Get test set metrics
r2, mse = get_test_metrics(rf, batch_size, n_batches_te, sample_type, exp, n_timesteps_in, n_timesteps_out, feature_set, remove_band)
print(f'Test metrics: R2 {r2}. MSE {mse}.') 


with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=f'test'):
    mlflow.log_param(f"n_batches_te", n_batches_te)
    mlflow.log_param("batch_size", batch_size)
    
# Get run that was just created and its ID to use when tracking
client = mlflow.tracking.MlflowClient() # Create a MlflowClient object
runs = client.search_runs(experiment.experiment_id)
mlflow_run_id_test = [r.info.run_id for r in runs if r.info.run_name==f'test'][0]

client.log_metric(mlflow_run_id_test,"r2", r2, step=epoch)
client.log_metric(mlflow_run_id_test,"mse", mse, step=epoch)



#jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/RF/RF_basic.stdout -err sj_out/RF/RF_basic.stderr python RF_basic.py