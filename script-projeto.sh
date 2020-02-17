##########################################################################
# In my repository: https://github.com/RafaelRampineli/Hadoop_Ecosystem/blob/master/Hadoop_Install/InstalandoHadoop_Parte1.txt 
you can find a step-by-step how to config a Full Hadoop Enviroment (hadoop, Hdfs, Spark, Hive, Hbase, Zookeeper, Pig, Scoop e Flume)
##########################################################################

# TOP Hive Commands: https://www.edureka.co/blog/hive-commands-with-examples

# Start Services Hadoop-Hdfs, Yarn and Hive Shell:
start-dfs.sh
start-yarn.sh
hive

##########################################################################
# Stage 1 - Loading a dataset to Hive and viewing data using Hive SQL
##########################################################################

# Using this commando you can see the database you're connected inside Hive:
SET hive.cli.print.current.db = true;

CREATE DATABASE project2; 

USE project2;

SHOW tables;

CREATE TABLE pacientes (ID INT, IDADE INT, SEXO INT, PRESSAO_SANGUINEA INT, COLESTEROL INT, ACUCAR_SANGUE INT, ECG INT, BATIMENTOS INT, DOENCA INT ) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE; 

LOAD DATA LOCAL INPATH '<Your_Local_directory>/pacientes.csv' OVERWRITE INTO TABLE pacientes;

SELECT count(*) FROM pacientes;

SELECT doenca, count(*), avg(idade), avg(pressao_sanguinea), avg(colesterol), avg(acucar_sangue), avg(batimentos) FROM pacientes GROUP BY doenca;


##########################################################################
# Stage 2 - Exploratory Analysis and pre-processing data using PIG
##########################################################################

# Create directory inside HDFS
hdfs dfs -mkdir /user/dados
hdfs dfs -mkdir /user/dados/pacientes

# Copy the file from Local to HDFS:
hdfs dfs -copyFromLocal /user/dados/pacientes/pacientes.csv /user/dados/pacientes

# Start Job History before using PIG:
mr-jobhistory-daemon.sh start historyserver

# Open Pig-shell multicluster mode:
pig -x mapreduce

# Loading data to PIG from HDFS Directory:
dadosPacientes = LOAD '/user/dados/pacientes/pacientes.csv' USING PigStorage(',') AS ( ID:int, Idade:int, Sexo:int, PressaoSanguinea:int, Colesterol:int, AcucarSangue:int, ECG:int, Batimentos:int, Doenca:int);

DUMP dadosPacientes

# Register an UDF.jar to use math operations
REGISTER <Diretório_local>/datafu-1.2.0.jar; 

# Creating an alias to re-use the UDF function many times so fast as possible:
DEFINE Quantile datafu.pig.stats.Quantile('0.0','0.25','0.5','0.75','1.0'); 

#  Assigning the result of group to a variable:
GrupoDoenca = GROUP dadosPacientes BY Doenca; 

DUMP GrupoDoenca;

# Making a Quartile analysis variables
quanData = FOREACH GrupoDoenca GENERATE group, Quantile(dadosPacientes.Idade) as Age, Quantile(dadosPacientes.PressaoSanguinea) as BP, Quantile(dadosPacientes.Colesterol) as Colesterol, Quantile(dadosPacientes.AcucarSangue) as AcucarSangue; 

DUMP quanData;

##########################################################################
# Stage 3 - Transforming Data using PIG
##########################################################################

# Scale standardization
IdadeRange = FOREACH dadosPacientes GENERATE ID, CEIL(Idade/10) as IdadeRange; 
bpRange = FOREACH dadosPacientes GENERATE ID, CEIL(PressaoSanguinea/25) as bpRange; 
chRange = FOREACH dadosPacientes GENERATE ID, CEIL(Colesterol/25) as chRange; 
hrRange = FOREACH dadosPacientes GENERATE ID, CEIL(Batimentos/25) as hrRange; 

DUMP IdadeRange;

# Join many results inside one variable
FullData = JOIN dadosPacientes by ID, IdadeRange by ID, bpRange by ID, hrRange by ID; 

describe FullData;

# Chossing the best variables to use as prediction of Machine Learning Model
predictionData = FOREACH FullData GENERATE dadosPacientes::Sexo, dadosPacientes::AcucarSangue, dadosPacientes::ECG, IdadeRange::IdadeRange, bpRange::bpRange, hrRange::hrRange, dadosPacientes::Doenca; 

STORE predictionData INTO '/user/dados/pacientes/DadosPacientes_Prediction' USING PigStorage(',');

##########################################################################
# Stage 4 - Classification Prediction Model

# Mahout version 0.13 doesn't have a RANDOM FOREST model
# you must downgrade version to Mahout 0.11
##########################################################################

# Just use this if your output stage 3 was in S.O.
hdfs dfs -mkdir //projeto_Preditivo_Mahout
hdfs dfs -copyFromLocal DadosPacientes_Prediction/* //projeto_Preditivo_Mahout

mahout describe -p /user/dados/pacientes/DadosPacientes_Prediction/part-r-00000 -f /user/dados/pacientes/DadosPacientes_Prediction/desc -d 6 N L

# Splitting Data in Train and Test
mahout splitDataset --input /user/dados/pacientes/DadosPacientes_Prediction/part-r-00000 --output /user/dados/pacientes/DadosPacientes_Prediction/splitdata --trainingPercentage 0.7 --probePercentage 0.3

# Build the RandomForest Model
mahout buildforest -d /user/dados/pacientes/DadosPacientes_Prediction/splitdata/trainingSet/* -ds /user/dados/pacientes/DadosPacientes_Prediction/desc -sl 3 -p -t 1 -o /user/dados/pacientes/DadosPacientes_Prediction/model

# Testing Model Result
mahout testforest -i /user/dados/pacientes/DadosPacientes_Prediction/splitdata/probeSet -ds /user/dados/pacientes/DadosPacientes_Prediction/desc -m /user/dados/pacientes/DadosPacientes_Prediction/model -a -mr -o /user/dados/pacientes/DadosPacientes_Prediction/predictions

# OutPut will be a confusion matrix

##########################################################################
# Stage 5 - Optimization Prediction Model

# Everytime when you increse tree numbers, the model accuracy increase.
##########################################################################

# Build the model using 25 tree to increase accuracy
mahout buildforest -d /user/dados/pacientes/DadosPacientes_Prediction/splitdata/trainingSet/* -ds /user/dados/pacientes/DadosPacientes_Prediction/desc -sl 3 -p -t 25 -o /user/dados/pacientes/DadosPacientes_Prediction/model

# Testing Model Result
mahout testforest -i /user/dados/pacientes/DadosPacientes_Prediction/splitdata/probeSet -ds /user/dados/pacientes/DadosPacientes_Prediction/desc -m /user/dados/pacientes/DadosPacientes_Prediction/model -a -mr -o /user/dados/pacientes/DadosPacientes_Prediction/predictions
