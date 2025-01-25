source ~/miniconda3/bin/activate

# conda create
conda create --name mlopsassign python=3.10 -y

conda activate mlopsassign


# pip upgrade
pip install --upgrade pip

# setup jupyter 
pip install ipykernel



Data link : https://www.kaggle.com/code/kamaljit/house-price-prediction/input


pip install dvc
pip install "dvc[azure]"



dvc init
dvc remote add 



dvc remote add -d azure_remote azure://version-model-container



az storage account show-connection-string --name datamodelgrp51 --resource-group mlops-group51
{
  "connectionString": "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=datamodelgrp51;AccountKey=TJqeNyqkmg1G4iyvljhoywCyvvfx8iI537ZgxMfFMgrSpiaRcapdDWiTuHtJYDo+udIS01/6IXoH+AStmRIwNQ==;BlobEndpoint=https://datamodelgrp51.blob.core.windows.net/;FileEndpoint=https://datamodelgrp51.file.core.windows.net/;QueueEndpoint=https://datamodelgrp51.queue.core.windows.net/;TableEndpoint=https://datamodelgrp51.table.core.windows.net/"
}



dvc remote modify azure_remote connection_string "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=datamodelgrp51;AccountKey=TJqeNyqkmg1G4iyvljhoywCyvvfx8iI537ZgxMfFMgrSpiaRcapdDWiTuHtJYDo+udIS01/6IXoH+AStmRIwNQ==;BlobEndpoint=https://datamodelgrp51.blob.core.windows.net/;FileEndpoint=https://datamodelgrp51.file.core.windows.net/;QueueEndpoint=https://datamodelgrp51.queue.core.windows.net/;TableEndpoint=https://datamodelgrp51.table.core.windows.net/"


dvc add data

git add data.dvc


dvc push

git log

git checkout b5235fcf0184d5244ced01ca9f63dc70bebcd3ed data.dvc

dvc pull

git commit -m "Revert to initial version of data.csv"





python models/ml-model.py 

mlflow ui --gunicorn-opts="--timeout 120"
mlflow ui --port 5001    







Blob SAS token

sp=racwdli&st=2025-01-23T14:42:55Z&se=2025-01-30T22:42:55Z&spr=https&sv=2022-11-02&sr=c&sig=PkAD6wZitZkzmYDQPwqyMp%2F1m%2FtZez3x9%2F1kSDUPiF4%3D

Blob SAS URL

https://datamodelgrp51.blob.core.windows.net/version-model-container?sp=racwdli&st=2025-01-23T14:42:55Z&se=2025-01-30T22:42:55Z&spr=https&sv=2022-11-02&sr=c&sig=PkAD6wZitZkzmYDQPwqyMp%2F1m%2FtZez3x9%2F1kSDUPiF4%3D



url : https://datamodelgrp51.blob.core.windows.net/version-model-container