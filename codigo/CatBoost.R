# CatBoost ---------------------------------------------------------------------


# Paqueterias ------------------------------------------------------------------
# Cómo instalar CatBoost
# https://catboost.ai/docs/en/installation/r-installation-binary-installation
# install.packages('remotes')
#library('remotes')
#remotes::install_url( "https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-R-windows-x86_64-1.2.10.tgz",INSTALL_opts = c("--no-multiarch", "--no-test-load"))

library(catboost)
library(tidyverse)
library(purrr)
library(dplyr)
library(rsample)
library(caret)
library(tictoc)

# Funciones Auxiliares ---------------------------------------------------------

source("codigo/FuncionesAuxiliares.R")


# Conjuntos Train y Test -------------------------------------------------------

load("datos/Folds.RData")

# CatBoost ---------------------------------------------------------------------

# Modelo Final -----------------------------------------------------------------

CATBOOST <- function(df, nombre_modelo, genesKP = "mezcla"){
  
  tic()
  set.seed(123)
  
  # Filtramos variables de interés
  
  if(genesKP == "grupo_1"){
    
    df_filtrado <- df[ , c(grupo_1, "clase")]
    
  } else if (genesKP == "grupo_2"){
    
    df_filtrado <- df[ , c(grupo_2, "clase")]
    
  } else if (genesKP == "grupo_3"){
    
    df_filtrado <- df[ , c(grupo_3, "clase")]
    
  } else if (genesKP == "mezcla") {
    
    df_filtrado <- df
    
  }
  
  
  # Matriz de predictores y variable objetivo
  
  X <- as.matrix(
    df_filtrado[
      , -which(colnames(df_filtrado) == "clase")
    ]
  )
  
  Y <- as.numeric(df_filtrado$clase) - 1
  
  
  # Pool CatBoost
  
  pool_data <- catboost.load_pool(
    data = X,
    label = Y
  )
  
  
  # Malla de hiperparametros
  
  malla <- expand.grid(
    iterations = c(250, 500),           # Numero maximo de arboles
    learning_rate = c(0.1),       # Tasa de aprendizaje
    depth = c(4, 6),                 # Profundidad de los arboles
    rsm = c(0.5, 0.8)                # Fracción de variables a considerar
  )
  
  
  # Parametros base
  
  parametros_base <- list(
    loss_function = "Logloss",
    eval_metric = "Logloss",
    
    boosting_type = "Ordered",
    grow_policy = "SymmetricTree",
    
    bootstrap_type = "Bernoulli",
    
    random_seed = 123,
    
    logging_level = "Silent"
  )
  
  
  # Inicializamos objetos
  
  mejor_error <- Inf
  mejores_parametros <- NULL
  mejor_iteracion <- NULL
  
  
  # Tuneo de hiperparametros
  
  for (i in 1:nrow(malla)) {
    
    # Parametros de esta combinacion
    
    parametros_actuales <- c(
      parametros_base,
      as.list(malla[i, ])
    )
    
    
    # Validacion cruzada
    
    cv_result <- catboost.cv(
      pool = pool_data,
      params = parametros_actuales,
      fold_count = 5,
      type = "Classical",
      partition_random_seed = 123,
      stratified = TRUE,
      early_stopping_rounds = 10
    )
    
    
    # Logloss promedio de validacion por iteracion
    
    metricas <- cv_result$test.Logloss.mean
    
    
    # Mejor iteracion
    
    mejor_iteracion_actual <- which.min(metricas)
    
    
    # Error correspondiente exactamente a esa iteracion
    
    mejor_error_actual <- metricas[mejor_iteracion_actual]
    
    
    # Guardamos la mejor combinacion
    
    if(mejor_error_actual < mejor_error){
      
      mejor_error <- mejor_error_actual
      
      mejores_parametros <- parametros_actuales
      
      mejor_iteracion <- mejor_iteracion_actual
      
    }
    
  }
  
  
  # Parametros del modelo final
  
  parametros_finales <- mejores_parametros
  
  parametros_finales$iterations <- mejor_iteracion
  
  
  # Modelo final
  
  catboost_final <- catboost.train(
    learn_pool = pool_data,
    params = parametros_finales
  )
  
  
  # Parametros que guardamos para reportar
  
  mejores_parametros_salida <- c(
    mejores_parametros,
    list(
      best_iteration = mejor_iteracion
    )
  )
  
  
  # Importancia de variables
  
  importancia <- catboost.get_feature_importance(
    model = catboost_final,
    pool = pool_data,
    type = "PredictionValuesChange"
  )
  
  
  importancia_df <- data.frame(
    Feature = colnames(X),
    Importance = importancia
  ) %>%
    arrange(desc(Importance)) %>%
    head(10)
  
  
  # Medidas aparentes
  
  predicciones <- catboost.predict(
    model = catboost_final,
    pool = pool_data,
    prediction_type = "Probability"
  )
  
  
  predicciones_clase <- ifelse(
    predicciones >= 0.5,
    levels(df_filtrado$clase)[2],
    levels(df_filtrado$clase)[1]
  )
  
  
  matriz <- confusionMatrix(
    data = factor(
      predicciones_clase,
      levels = levels(df_filtrado$clase)
    ),
    reference = df_filtrado$clase,
    positive = levels(df_filtrado$clase)[2]
  )
  
  
  respond <- data.frame(
    Metric = c(
      "accuracy",
      "recall",
      "specificity"
    ),
    Score = c(
      as.numeric(matriz$overall["Accuracy"]),
      as.numeric(matriz$byClass["Sensitivity"]),
      as.numeric(matriz$byClass["Specificity"])
    )
  )
  
  
  # Tiempo de ejecucion
  
  tiempo <- toc(quiet = TRUE)
  
  
  # Resultados
  
  return(list(
    nombreModelo = nombre_modelo,
    metricasAparentes = respond,
    matrizConfusion = matriz$table,
    importancia = importancia_df,
    modelo = catboost_final,
    mejores_parametros = mejores_parametros_salida,
    tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
  
}

# Cálculo del poder predictivo -------------------------------------------------

CATBOOST.PP <- function(folds, nombre_modelo, genesKP = "mezcla"){
  
  tic()
  set.seed(123)
  
  
  modCatBoost <- function(split){
    
    # Train y Test
    
    train_data <- analysis(split)
    test_data  <- assessment(split)
    
    
    # Filtramos por las variables de interés
    
    if(genesKP == "grupo_1"){
      
      train_data_filtrado <- train_data[ , c(grupo_1, "clase")]
      test_data_filtrado  <- test_data[ , c(grupo_1, "clase")]
      
    } else if (genesKP == "grupo_2"){
      
      train_data_filtrado <- train_data[ , c(grupo_2, "clase")]
      test_data_filtrado  <- test_data[ , c(grupo_2, "clase")]
      
    } else if (genesKP == "grupo_3"){
      
      train_data_filtrado <- train_data[ , c(grupo_3, "clase")]
      test_data_filtrado  <- test_data[ , c(grupo_3, "clase")]
      
    } else if (genesKP == "mezcla"){
      
      train_data_filtrado <- train_data
      test_data_filtrado  <- test_data
      
    }
    
    
    # Matrices de entrenamiento y prueba
    
    X_train <- as.matrix(
      train_data_filtrado[
        , -which(colnames(train_data_filtrado) == "clase")
      ]
    )
    
    Y_train <- as.numeric(train_data_filtrado$clase) - 1
    
    
    X_test <- as.matrix(
      test_data_filtrado[
        , -which(colnames(test_data_filtrado) == "clase")
      ]
    )
    
    Y_test <- test_data_filtrado$clase
    
    
    # Pools CatBoost
    
    train_pool <- catboost.load_pool(
      data = X_train,
      label = Y_train
    )
    
    test_pool <- catboost.load_pool(
      data = X_test
    )
    
    
    # Malla de hiperparametros
    
    malla <- expand.grid(
      iterations = c(250, 500),           # Numero maximo de arboles
      learning_rate = c( 0.1),       # Tasa de aprendizaje
      depth = c(4, 6),                 # Profundidad de los arboles
      rsm = c(0.5, 0.8)                # Fracción de variables a considerar
    )
    
    
    # Parametros base
    
    parametros_base <- list(
      loss_function = "Logloss",
      eval_metric = "Logloss",
      
      boosting_type = "Ordered",
      grow_policy = "SymmetricTree",
      
      bootstrap_type = "Bernoulli",
      
      random_seed = 123,
      
      logging_level = "Silent"
    )
    
    
    # Inicializamos objetos
    
    mejor_error <- Inf
    mejores_parametros <- NULL
    mejor_iteracion <- NULL
    
    
    # Tuneo de hiperparametros
    
    for (i in 1:nrow(malla)) {
      
      # Parametros de esta combinacion
      
      parametros_actuales <- c(
        parametros_base,
        as.list(malla[i, ])
      )
      
      
      # Validacion cruzada interna
      
      cv_result <- catboost.cv(
        pool = train_pool,
        params = parametros_actuales,
        fold_count = 5,
        type = "Classical",
        partition_random_seed = 123,
        stratified = TRUE,
        early_stopping_rounds = 10
      )
      
      
      # Logloss promedio de validacion por iteracion
      
      metricas <- cv_result$test.Logloss.mean
      
      
      # Mejor iteracion
      
      mejor_iteracion_actual <- which.min(metricas)
      
      
      # Error correspondiente exactamente a esa iteracion
      
      mejor_error_actual <- metricas[mejor_iteracion_actual]
      
      
      # Guardamos la mejor combinacion
      
      if(mejor_error_actual < mejor_error){
        
        mejor_error <- mejor_error_actual
        
        mejores_parametros <- parametros_actuales
        
        mejor_iteracion <- mejor_iteracion_actual
        
      }
      
    }
    
    
    # Parametros del modelo final del fold
    
    parametros_finales <- mejores_parametros
    
    parametros_finales$iterations <- mejor_iteracion
    
    
    # Modelo final del fold
    
    catboost_final <- catboost.train(
      learn_pool = train_pool,
      params = parametros_finales
    )
    
    
    # Predicciones sobre conjunto de prueba
    
    predicciones <- catboost.predict(
      model = catboost_final,
      pool = test_pool,
      prediction_type = "Probability"
    )
    
    
    predicciones_clase <- ifelse(
      predicciones >= 0.5,
      levels(train_data_filtrado$clase)[2],
      levels(train_data_filtrado$clase)[1]
    )
    
    
    # Matriz de confusion
    
    matriz <- confusionMatrix(
      data = factor(
        predicciones_clase,
        levels = levels(train_data_filtrado$clase)
      ),
      reference = Y_test,
      positive = levels(train_data_filtrado$clase)[2]
    )
    
    
    # Metricas
    
    respond <- data.frame(
      Metric = c(
        "accuracy",
        "recall",
        "specificity"
      ),
      Score = c(
        as.numeric(matriz$overall["Accuracy"]),
        as.numeric(matriz$byClass["Sensitivity"]),
        as.numeric(matriz$byClass["Specificity"])
      )
    )
    
    
    return(respond[, 2])
    
  }
  
  
  # Ejecutamos el modelo en cada fold
  
  resultados_folds <- lapply(
    folds$splits,
    modCatBoost
  )
  
  
  # Promedio de metricas
  
  promedio_metricas <- rowMeans(
    do.call(cbind, resultados_folds)
  )
  
  
  resultados <- data.frame(
    metricas = c(
      "accuracy",
      "recall",
      "specificity"
    ),
    promedio = promedio_metricas
  )
  
  
  # Tiempo de ejecucion
  
  tiempo <- toc(quiet = TRUE)
  
  
  # Resultados
  
  return(list(
    nombreModelo = nombre_modelo,
    resultados = resultados,
    tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
  
}

# CatBoost ---------------------------------------------------------------------

# GRUPO 1 ----------------------------------------------------------------------

# BRAIN

M7.CATB.Brain.Grupo1.Final <- CATBOOST(
  df = datos_brain,
  nombre_modelo = "M7 CATB Brain Grupo1 Final",
  genesKP = "grupo_1"
)

M7.CATB.Brain.Grupo1.PP <- CATBOOST.PP(
  folds = folds_brain,
  nombre_modelo = "M7 CATB Brain Grupo1 PP",
  genesKP = "grupo_1"
)


# SKIN

M7.CATB.Skin.Grupo1.Final <- CATBOOST(
  df = datos_skin,
  nombre_modelo = "M7 CATB Skin Grupo1 Final",
  genesKP = "grupo_1"
)

M7.CATB.Skin.Grupo1.PP <- CATBOOST.PP(
  folds = folds_skin,
  nombre_modelo = "M7 CATB Skin Grupo1 PP",
  genesKP = "grupo_1"
)


# LUNG

M7.CATB.Lung.Grupo1.Final <- CATBOOST(
  df = datos_lung,
  nombre_modelo = "M7 CATB Lung Grupo1 Final",
  genesKP = "grupo_1"
)

M7.CATB.Lung.Grupo1.PP <- CATBOOST.PP(
  folds = folds_lung,
  nombre_modelo = "M7 CATB Lung Grupo1 PP",
  genesKP = "grupo_1"
)


# STOMACH

M7.CATB.Stomach.Grupo1.Final <- CATBOOST(
  df = datos_stomach,
  nombre_modelo = "M7 CATB Stomach Grupo1 Final",
  genesKP = "grupo_1"
)

M7.CATB.Stomach.Grupo1.PP <- CATBOOST.PP(
  folds = folds_stomach,
  nombre_modelo = "M7 CATB Stomach Grupo1 PP",
  genesKP = "grupo_1"
)


# THYROID

M7.CATB.Thyroid.Grupo1.Final <- CATBOOST(
  df = datos_thyroid,
  nombre_modelo = "M7 CATB Thyroid Grupo1 Final",
  genesKP = "grupo_1"
)

M7.CATB.Thyroid.Grupo1.PP <- CATBOOST.PP(
  folds = folds_thyroid,
  nombre_modelo = "M7 CATB Thyroid Grupo1 PP",
  genesKP = "grupo_1"
)


# BREAST

M7.CATB.Breast.Grupo1.Final <- CATBOOST(
  df = datos_breast,
  nombre_modelo = "M7 CATB Breast Grupo1 Final",
  genesKP = "grupo_1"
)

M7.CATB.Breast.Grupo1.PP <- CATBOOST.PP(
  folds = folds_breast,
  nombre_modelo = "M7 CATB Breast Grupo1 PP",
  genesKP = "grupo_1"
)



resultados_catb_grupo1 <- list(
  
  # BRAIN
  
  M7.CATB.Brain.Grupo1.Final = M7.CATB.Brain.Grupo1.Final,
  M7.CATB.Brain.Grupo1.PP    = M7.CATB.Brain.Grupo1.PP,
  
  
  # SKIN
  
  M7.CATB.Skin.Grupo1.Final = M7.CATB.Skin.Grupo1.Final,
  M7.CATB.Skin.Grupo1.PP    = M7.CATB.Skin.Grupo1.PP,
  
  
  # LUNG
  
  M7.CATB.Lung.Grupo1.Final = M7.CATB.Lung.Grupo1.Final,
  M7.CATB.Lung.Grupo1.PP    = M7.CATB.Lung.Grupo1.PP,
  
  
  # STOMACH
  
  M7.CATB.Stomach.Grupo1.Final = M7.CATB.Stomach.Grupo1.Final,
  M7.CATB.Stomach.Grupo1.PP    = M7.CATB.Stomach.Grupo1.PP,
  
  
  # THYROID
  
  M7.CATB.Thyroid.Grupo1.Final = M7.CATB.Thyroid.Grupo1.Final,
  M7.CATB.Thyroid.Grupo1.PP    = M7.CATB.Thyroid.Grupo1.PP,
  
  
  # BREAST
  
  M7.CATB.Breast.Grupo1.Final = M7.CATB.Breast.Grupo1.Final,
  M7.CATB.Breast.Grupo1.PP    = M7.CATB.Breast.Grupo1.PP
)


save(
  resultados_catb_grupo1,
  file = "resultados/resultados_CatBoost_grupo1.RData"
)


# GRUPO 2 ----------------------------------------------------------------------

# BRAIN

M7.CATB.Brain.Grupo2.Final <- CATBOOST(
  df = datos_brain,
  nombre_modelo = "M7 CATB Brain Grupo2 Final",
  genesKP = "grupo_2"
)

M7.CATB.Brain.Grupo2.PP <- CATBOOST.PP(
  folds = folds_brain,
  nombre_modelo = "M7 CATB Brain Grupo2 PP",
  genesKP = "grupo_2"
)


# SKIN

M7.CATB.Skin.Grupo2.Final <- CATBOOST(
  df = datos_skin,
  nombre_modelo = "M7 CATB Skin Grupo2 Final",
  genesKP = "grupo_2"
)

M7.CATB.Skin.Grupo2.PP <- CATBOOST.PP(
  folds = folds_skin,
  nombre_modelo = "M7 CATB Skin Grupo2 PP",
  genesKP = "grupo_2"
)


# LUNG

M7.CATB.Lung.Grupo2.Final <- CATBOOST(
  df = datos_lung,
  nombre_modelo = "M7 CATB Lung Grupo2 Final",
  genesKP = "grupo_2"
)

M7.CATB.Lung.Grupo2.PP <- CATBOOST.PP(
  folds = folds_lung,
  nombre_modelo = "M7 CATB Lung Grupo2 PP",
  genesKP = "grupo_2"
)


# STOMACH

M7.CATB.Stomach.Grupo2.Final <- CATBOOST(
  df = datos_stomach,
  nombre_modelo = "M7 CATB Stomach Grupo2 Final",
  genesKP = "grupo_2"
)

`
M7.CATB.Stomach.Grupo2.PP <- CATBOOST.PP(
  folds = folds_stomach,
  nombre_modelo = "M7 CATB Stomach Grupo2 PP",
  genesKP = "grupo_2"
)


# THYROID

M7.CATB.Thyroid.Grupo2.Final <- CATBOOST(
  df = datos_thyroid,
  nombre_modelo = "M7 CATB Thyroid Grupo2 Final",
  genesKP = "grupo_2"
)

M7.CATB.Thyroid.Grupo2.PP <- CATBOOST.PP(
  folds = folds_thyroid,
  nombre_modelo = "M7 CATB Thyroid Grupo2 PP",
  genesKP = "grupo_2"
)


# BREAST

M7.CATB.Breast.Grupo2.Final <- CATBOOST(
  df = datos_breast,
  nombre_modelo = "M7 CATB Breast Grupo2 Final",
  genesKP = "grupo_2"
)

M7.CATB.Breast.Grupo2.PP <- CATBOOST.PP(
  folds = folds_breast,
  nombre_modelo = "M7 CATB Breast Grupo2 PP",
  genesKP = "grupo_2"
)


# Guardamos resultados

resultados_catb_grupo2 <- list(
  
  # BRAIN
  
  M7.CATB.Brain.Grupo2.Final = M7.CATB.Brain.Grupo2.Final,
  M7.CATB.Brain.Grupo2.PP    = M7.CATB.Brain.Grupo2.PP,
  
  
  # SKIN
  
  M7.CATB.Skin.Grupo2.Final = M7.CATB.Skin.Grupo2.Final,
  M7.CATB.Skin.Grupo2.PP    = M7.CATB.Skin.Grupo2.PP,
  
  
  # LUNG
  
  M7.CATB.Lung.Grupo2.Final = M7.CATB.Lung.Grupo2.Final,
  M7.CATB.Lung.Grupo2.PP    = M7.CATB.Lung.Grupo2.PP,
  
  
  # STOMACH
  
  M7.CATB.Stomach.Grupo2.Final = M7.CATB.Stomach.Grupo2.Final,
  M7.CATB.Stomach.Grupo2.PP    = M7.CATB.Stomach.Grupo2.PP,
  
  
  # THYROID
  
  M7.CATB.Thyroid.Grupo2.Final = M7.CATB.Thyroid.Grupo2.Final,
  M7.CATB.Thyroid.Grupo2.PP    = M7.CATB.Thyroid.Grupo2.PP,
  
  
  # BREAST
  
  M7.CATB.Breast.Grupo2.Final = M7.CATB.Breast.Grupo2.Final,
  M7.CATB.Breast.Grupo2.PP    = M7.CATB.Breast.Grupo2.PP
)


save(
  resultados_catb_grupo2,
  file = "resultados/resultados_CatBoost_grupo2.RData"
)

# GRUPO 3 ----------------------------------------------------------------------

# BRAIN

M7.CATB.Brain.Grupo3.Final <- CATBOOST(
  df = datos_brain,
  nombre_modelo = "M7 CATB Brain Grupo3 Final",
  genesKP = "grupo_3"
)

M7.CATB.Brain.Grupo3.PP <- CATBOOST.PP(
  folds = folds_brain,
  nombre_modelo = "M7 CATB Brain Grupo3 PP",
  genesKP = "grupo_3"
)


# SKIN

M7.CATB.Skin.Grupo3.Final <- CATBOOST(
  df = datos_skin,
  nombre_modelo = "M7 CATB Skin Grupo3 Final",
  genesKP = "grupo_3"
)

M7.CATB.Skin.Grupo3.PP <- CATBOOST.PP(
  folds = folds_skin,
  nombre_modelo = "M7 CATB Skin Grupo3 PP",
  genesKP = "grupo_3"
)


# LUNG

M7.CATB.Lung.Grupo3.Final <- CATBOOST(
  df = datos_lung,
  nombre_modelo = "M7 CATB Lung Grupo3 Final",
  genesKP = "grupo_3"
)

M7.CATB.Lung.Grupo3.PP <- CATBOOST.PP(
  folds = folds_lung,
  nombre_modelo = "M7 CATB Lung Grupo3 PP",
  genesKP = "grupo_3"
)


# STOMACH

M7.CATB.Stomach.Grupo3.Final <- CATBOOST(
  df = datos_stomach,
  nombre_modelo = "M7 CATB Stomach Grupo3 Final",
  genesKP = "grupo_3"
)

M7.CATB.Stomach.Grupo3.PP <- CATBOOST.PP(
  folds = folds_stomach,
  nombre_modelo = "M7 CATB Stomach Grupo3 PP",
  genesKP = "grupo_3"
)


# THYROID

M7.CATB.Thyroid.Grupo3.Final <- CATBOOST(
  df = datos_thyroid,
  nombre_modelo = "M7 CATB Thyroid Grupo3 Final",
  genesKP = "grupo_3"
)

M7.CATB.Thyroid.Grupo3.PP <- CATBOOST.PP(
  folds = folds_thyroid,
  nombre_modelo = "M7 CATB Thyroid Grupo3 PP",
  genesKP = "grupo_3"
)


# BREAST

M7.CATB.Breast.Grupo3.Final <- CATBOOST(
  df = datos_breast,
  nombre_modelo = "M7 CATB Breast Grupo3 Final",
  genesKP = "grupo_3"
)

M7.CATB.Breast.Grupo3.PP <- CATBOOST.PP(
  folds = folds_breast,
  nombre_modelo = "M7 CATB Breast Grupo3 PP",
  genesKP = "grupo_3"
)


# Guardamos resultados

resultados_catb_grupo3 <- list(
  
  # BRAIN
  
  M7.CATB.Brain.Grupo3.Final = M7.CATB.Brain.Grupo3.Final,
  M7.CATB.Brain.Grupo3.PP    = M7.CATB.Brain.Grupo3.PP,
  
  
  # SKIN
  
  M7.CATB.Skin.Grupo3.Final = M7.CATB.Skin.Grupo3.Final,
  M7.CATB.Skin.Grupo3.PP    = M7.CATB.Skin.Grupo3.PP,
  
  
  # LUNG
  
  M7.CATB.Lung.Grupo3.Final = M7.CATB.Lung.Grupo3.Final,
  M7.CATB.Lung.Grupo3.PP    = M7.CATB.Lung.Grupo3.PP,
  
  
  # STOMACH
  
  M7.CATB.Stomach.Grupo3.Final = M7.CATB.Stomach.Grupo3.Final,
  M7.CATB.Stomach.Grupo3.PP    = M7.CATB.Stomach.Grupo3.PP,
  
  
  # THYROID
  
  M7.CATB.Thyroid.Grupo3.Final = M7.CATB.Thyroid.Grupo3.Final,
  M7.CATB.Thyroid.Grupo3.PP    = M7.CATB.Thyroid.Grupo3.PP,
  
  
  # BREAST
  
  M7.CATB.Breast.Grupo3.Final = M7.CATB.Breast.Grupo3.Final,
  M7.CATB.Breast.Grupo3.PP    = M7.CATB.Breast.Grupo3.PP
)


save(
  resultados_catb_grupo3,
  file = "resultados/resultados_CatBoost_grupo3.RData"
)
