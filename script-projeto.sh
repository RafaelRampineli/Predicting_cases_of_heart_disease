##########################################################################
# para saber como configurar um ambiente Hadoop completo (hadoop, Hdfs, Spark, Hive, Hbase, Zookeeper, Pig, Scoop e Flume),  Acesse: https://github.com/RafaelRampineli/Hadoop_Ecosystem/blob/master/Hadoop_Install/InstalandoHadoop_Parte1.txt
##########################################################################

# TOP Hive Commands: https://www.edureka.co/blog/hive-commands-with-examples

# Projeto - Prevendo Doencas Cardiacas

# Start Servi�os:
start-dfs.sh
start-yarn.sh
hive

##########################################################################
# Etapa 1 - Carregando o dataset no Hive e visualizando os dados com SQL
##########################################################################

# Comando para monstrar a qual database est� conectado dentro do Hive:
SET hive.cli.print.current.db = true;

CREATE DATABASE project2; 

USE project2;

SHOW tables;

CREATE TABLE pacientes (ID INT, IDADE INT, SEXO INT, PRESSAO_SANGUINEA INT, COLESTEROL INT, ACUCAR_SANGUE INT, ECG INT, BATIMENTOS INT, DOENCA INT ) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE; 

LOAD DATA LOCAL INPATH '<Diret�rio_local>/pacientes.csv' OVERWRITE INTO TABLE pacientes;

SELECT count(*) FROM pacientes;

SELECT doenca, count(*), avg(idade), avg(pressao_sanguinea), avg(colesterol), avg(acucar_sangue), avg(batimentos) FROM pacientes GROUP BY doenca;


##########################################################################
# Etapa 2 - Analise Exploratoria e pre-processamento nos dados com Pig
##########################################################################

# Criar diret�rio
hdfs dfs -mkdir /user/dados
hdfs dfs -mkdir /user/dados/pacientes

# Copiar o Arquivo para dentro do HDFS:
hdfs dfs -copyFromLocal /user/dados/pacientes/pacientes.csv /user/dados/pacientes

# Antes de executar opera��es com o PIG � necess�rio start no daemon:
# Inicie o Job History Server
mr-jobhistory-daemon.sh start historyserver

# Abrir o Shell do Pig:
pig -x mapreduce

# Carregar os dados do diret�rio HDFS
dadosPacientes = LOAD '/user/dados/pacientes/pacientes.csv' USING PigStorage(',') AS ( ID:int, Idade:int, Sexo:int, PressaoSanguinea:int, Colesterol:int, AcucarSangue:int, ECG:int, Batimentos:int, Doenca:int);

DUMP dadosPacientes

# Registrar a UDF.jar para ser poss�vel realiazar opera��es matem�ticas abaixo
REGISTER <Diret�rio_local>/datafu-1.2.0.jar; 

# Criando um alias para uma fun��o da UDF para utiliza��es futuras
DEFINE Quantile datafu.pig.stats.Quantile('0.0','0.25','0.5','0.75','1.0'); 

# Atribuindo o resultado de um agrupamento de Todos Dados dos pacientes por Doen�a a variavel GrupoDoenca
GrupoDoenca = GROUP dadosPacientes BY Doenca; 

DUMP GrupoDoenca;

# Ap�s o agrupamento, � realizado a an�lise dos Quartis de cada vari�vel.
# Os dados ser�o representados por : Doen�a como chaves e os quartils como dados;
quanData = FOREACH GrupoDoenca GENERATE group, Quantile(dadosPacientes.Idade) as Age, Quantile(dadosPacientes.PressaoSanguinea) as BP, Quantile(dadosPacientes.Colesterol) as Colesterol, Quantile(dadosPacientes.AcucarSangue) as AcucarSangue; 

# Mostra os Dados Carregados
DUMP quanData;

##########################################################################
# Etapa 3 - Transforma��o de Dados com o Pig
##########################################################################

# Realizando uma padroniza��o de escala nos dados.
IdadeRange = FOREACH dadosPacientes GENERATE ID, CEIL(Idade/10) as IdadeRange; 
bpRange = FOREACH dadosPacientes GENERATE ID, CEIL(PressaoSanguinea/25) as bpRange; 
chRange = FOREACH dadosPacientes GENERATE ID, CEIL(Colesterol/25) as chRange; 
hrRange = FOREACH dadosPacientes GENERATE ID, CEIL(Batimentos/25) as hrRange; 

DUMP IdadeRange;

# Realizando a jun��o dos novos dados padronizados ao dataset original
FullData = JOIN dadosPacientes by ID, IdadeRange by ID, bpRange by ID, hrRange by ID; 

describe FullData;

# Escolhendo as vari�veis que ser�o utilizadas como Preditoras do Modelo de ML
predictionData = FOREACH FullData GENERATE dadosPacientes::Sexo, dadosPacientes::AcucarSangue, dadosPacientes::ECG, IdadeRange::IdadeRange, bpRange::bpRange, hrRange::hrRange, dadosPacientes::Doenca; 

STORE predictionData INTO '/user/dados/pacientes/DadosPacientes_Prediction' USING PigStorage(',');


##########################################################################
# Etapa 4 - Cria��o do Modelo Preditivo de Classifica��o

# Vers�o Mahout 0.13 n�o contem modelo de classifica��o RANDOM FOREST
# Necess�rio utilizar Mahout vers�o 0.11
##########################################################################

# Caso os dados tenha sido salvo no S.O., Copia o arquivo gerado pela transforma��o com o Pig para o HDFS 
hdfs dfs -mkdir //projeto_Preditivo_Mahout
hdfs dfs -copyFromLocal DadosPacientes_Prediction/* //projeto_Preditivo_Mahout

# Cria um descritor para os dados
mahout describe -p /user/dados/pacientes/DadosPacientes_Prediction/part-r-00000 -f /user/dados/pacientes/DadosPacientes_Prediction/desc -d 6 N L

# Divide os dados em treino e teste 
mahout splitDataset --input /user/dados/pacientes/DadosPacientes_Prediction/part-r-00000 --output /user/dados/pacientes/DadosPacientes_Prediction/splitdata --trainingPercentage 0.7 --probePercentage 0.3

# Constr�i o modelo RandomForest com uma �rvore 
mahout buildforest -d /user/dados/pacientes/DadosPacientes_Prediction/splitdata/trainingSet/* -ds /user/dados/pacientes/DadosPacientes_Prediction/desc -sl 3 -p -t 1 -o /user/dados/pacientes/DadosPacientes_Prediction/model

# Testa o modelo
mahout testforest -i /user/dados/pacientes/DadosPacientes_Prediction/splitdata/probeSet -ds /user/dados/pacientes/DadosPacientes_Prediction/desc -m /user/dados/pacientes/DadosPacientes_Prediction/model -a -mr -o /user/dados/pacientes/DadosPacientes_Prediction/predictions

# OutPut ser� uma confusion matrix com os dados treinados e teste de acur�cia do modelo

##########################################################################
# Etapa 5 - Otimiza��o do Modelo Preditivo de Classifica��o

# Ao aumentar o n�mero de �rvores utilizado pelo modelo, a taxa de acur�cia aumenta.
##########################################################################

# Construir o modelo com 25 �rvores, a fim de aumentar a acur�cia
mahout buildforest -d /user/dados/pacientes/DadosPacientes_Prediction/splitdata/trainingSet/* -ds /user/dados/pacientes/DadosPacientes_Prediction/desc -sl 3 -p -t 25 -o /user/dados/pacientes/DadosPacientes_Prediction/model

# Testa o modelo
mahout testforest -i /user/dados/pacientes/DadosPacientes_Prediction/splitdata/probeSet -ds /user/dados/pacientes/DadosPacientes_Prediction/desc -m /user/dados/pacientes/DadosPacientes_Prediction/model -a -mr -o /user/dados/pacientes/DadosPacientes_Prediction/predictions
