import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import accuracy_score

def criaRede (  ) :
    """

    Função que faz a rede neural e a
    retorna

    :return:
    """

    Regressor = Sequential()

    Regressor.add ( Dense(
        units = 158,
        activation = "relu",
        input_dim = 316)
    )

    Regressor.add ( Dense(
        units = 158,
        activation = "relu")
    )
    Regressor.add (Dense(
        units=1,
        activation = 'linear')
    )

    Regressor.compile (
        loss = "mean_absolute_error",
        optimizer='adam',
        metrics = ['mean_absolute_error']
    )

    return Regressor

def processamentoDatabase (  ) :
    """
    Função que faz o pre - processamento
    do database e o retorna
    sem atributos desnecessários

    :return: database
    """

    df = pd.read_csv("autos.csv", encoding="ISO-8859-1")

    #return saidaPrevisao, saidaReal

    # apagando as colunas desnecessárias
    df = df.drop("dateCrawled", axis = 1 )
    df = df.drop("nrOfPictures", axis = 1 )
    df = df.drop("dateCreated", axis = 1)
    df = df.drop("postalCode", axis = 1 )
    df = df.drop("lastSeen", axis = 1 )
    df = df.drop("name", axis = 1 )
    df = df.drop("seller", axis = 1 )
    df = df.drop("offerType", axis = 1 )

    # apagando os registros incosistentes
    df = df [ df.price > 100 ]
    df = df.loc [ df.price < 350000 ]

    # obtendo o veículo que mais aparece
    # e substituindo os dados NaN no lugar
    # onde mais ele aparece
    print ( df["vehicleType"].value_counts (  ) )   # limousine     77933
    print ("\n----------------\n")
    print ( df["gearbox"].value_counts (  ) )   # manuell
    print ("\n----------------\n")
    print ( df["model"].value_counts (  ) )     # golf
    print ("\n----------------\n")
    print ( df["notRepairedDamage"].value_counts (  ) )     # nein
    print ("\n----------------\n")
    print ( df["fuelType"].value_counts (  ) )  #benzin

    # fazendo a mudança dos dados
    mudancaValores = {
        "vehicleType" : "limousine",
        "gearbox" : "manuell",
        "model" : "golf",
        "notRepairedDamage" : "nein",
        "fuelType" : "benzin"
    }

    df = df.fillna( value = mudancaValores )

    saidaPrevisao = df.iloc[:, 1:13].values
    saidaReal = df.iloc[:, 0].values

    labelEnconderPrevisores = LabelEncoder()

    saidaPrevisao[:, 0] = labelEnconderPrevisores.fit_transform ( saidaPrevisao [:, 0] )
    saidaPrevisao[:, 1] = labelEnconderPrevisores.fit_transform ( saidaPrevisao [:, 1] )
    saidaPrevisao[:, 3] = labelEnconderPrevisores.fit_transform ( saidaPrevisao [:, 3] )
    saidaPrevisao[:, 5] = labelEnconderPrevisores.fit_transform ( saidaPrevisao [:, 5] )
    saidaPrevisao[:, 8] = labelEnconderPrevisores.fit_transform ( saidaPrevisao [:, 8] )
    saidaPrevisao[:, 9] = labelEnconderPrevisores.fit_transform ( saidaPrevisao [:, 9] )
    saidaPrevisao[:, 10] = labelEnconderPrevisores.fit_transform ( saidaPrevisao[:, 10] )

    onehotencoder = OneHotEncoder ( categorical_features = [0, 1, 3, 5, 8, 9, 10] )
    saidaPrevisao = onehotencoder.fit_transform ( saidaPrevisao ).toarray()

    return saidaPrevisao, saidaReal

def main (  ) :

    X, Y = processamentoDatabase (  )

    dadosEntradaTreinamento, dadosEntradaTeste, dadosSaidaTreinamento, dadosSaidaTeste = train_test_split(
        X,
        Y,
        test_size=0.25
    )

    Regressor = KerasRegressor ( build_fn = criaRede,
                                 epochs = 200,
                                 batch_size = 300)

    #print(sorted ( sklearn.metrics.SCORERS.keys() ) )

    #kfold = KFold(n_splits=10, random_state = 1)

    resultados = cross_val_score(estimator = Regressor,
                                 X = dadosEntradaTreinamento,
                                 y = dadosSaidaTreinamento,
                                 cv = 10)
    Regressor.fit ( dadosEntradaTreinamento, dadosSaidaTreinamento )
    predicao = Regressor.predict ( dadosEntradaTeste )

    plt.plot ( dadosSaidaTeste, "rs" )
    plt.plot ( predicao, "bs" )
    plt.title("gŕafico de análise")
    plt.grid(True)
    plt.show()

    print("Média do valor dos automóveis, em euros : {}\n".format(resultados.mean (  ) ) )
    print("Desvio Padrão : {}\n".format ( resultados.std (  ) ) )
    print("scoring do modelo : {}".format ( accuracy_score ( dadosSaidaTeste,
                                                             predicao) ) )

if __name__ == '__main__':

    main (  )