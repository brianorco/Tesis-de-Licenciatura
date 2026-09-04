# LightGBM ---------------------------------------------------------------------


# Paqueterias ------------------------------------------------------------------

#install.packages("lightgbm")
library(lightgbm)
library(tidyverse)
library(purrr)
library(plyr)
library(dplyr)
library(tidyr)
library(forcats)
library(e1071)
library(rsample)
library(caret)
library(tictoc)


# Funciones Auxiliares ---------------------------------------------------------

source("codigo/FuncionesAuxiliares.R")


# Conjuntos Train y Test -------------------------------------------------------

load("datos/Folds.RData")

# LightGBM  --------------------------------------------------------------------


# Modelo Final -----------------------------------------------------------------


LGBM <- function(df, nombre_modelo, genesKP = "mezcla"){
  
  tic()
  set.seed(123)
  
  # Filtramos variables de interes
  
  if(genesKP == "grupo_1"){
    df_filtrado <- df[ , c(grupo_1, "clase")]
  } else if (genesKP == "grupo_2"){
    df_filtrado <- df[ , c(grupo_2, "clase")]
  } else if (genesKP == "grupo_3"){
    df_filtrado <- df[ , c(grupo_3, "clase")]
  } else if (genesKP == "mezcla") {
    df_filtrado <- df
  }
  
  X <- as.matrix(
    df_filtrado[ , -which(colnames(df_filtrado) == "clase")]
  )
  
  Y <- as.numeric(df_filtrado$clase) - 1
  
  datos_lgbm <- lgb.Dataset(
    data = X,
    label = Y
  )
  
  
  # Malla de hiperparametros
  
  malla <- expand.grid(
    num_iterations = c(250, 500),      # Numero maximo de iteraciones
    learning_rate = c(0.01, 0.1),      # Tasa de aprendizaje
    num_leaves = c(4, 8, 16),          # Numero maximo de hojas
    top_rate = c(0.2, 0.3),            # Proporcion de gradientes grandes
    other_rate = c(0.1, 0.2),          # Proporcion de gradientes pequeños
    min_data_in_leaf = c(10, 20)       # Minimo de observaciones por hoja
  )
  
  
  # Parametros base
  
  parametros_base <- list(
    objective = "binary",
    metric = "binary_logloss",
    data_sample_strategy = "goss",
    verbosity = -1,
    feature_pre_filter = FALSE,
    seed = 123
  )
  
  
  mejor_error <- Inf
  mejores_parametros <- NULL
  mejor_iteracion <- NULL
  mejor_num_iterations <- NULL
  
  
  # Tuneo de hiperparametros
  
  for (i in 1:nrow(malla)) {
    
    # Numero maximo de iteraciones para esta combinacion
    num_iterations_actual <- malla$num_iterations[i]
    
    
    # Construimos los parametros sin incluir num_iterations
    # porque este se pasa mediante nrounds
    
    parametros_actuales <- c(
      parametros_base,
      as.list(malla[i,setdiff(names(malla), "num_iterations")])
    )
    
    
    # Validacion cruzada
    
    lgbm_tune <- lgb.cv(
      params = parametros_actuales,
      data = datos_lgbm,
      nrounds = num_iterations_actual,
      nfold = 5,
      early_stopping_rounds = 10,
      verbose = -1
    )
    
    
    # Metricas obtenidas en validacion cruzada
    
    metricas <- unlist(
      lgbm_tune$record_evals$valid$binary_logloss$eval
    )
    
    
    # Mejor iteracion encontrada por early stopping
    
    if(!is.null(lgbm_tune$best_iter) &&
       lgbm_tune$best_iter > 0){
      
      mejor_iteracion_actual <- lgbm_tune$best_iter
      
    } else {
      
      # En caso de que no se active early stopping
      mejor_iteracion_actual <- which.min(metricas)
      
    }
    
    
    # Error correspondiente exactamente a esa iteracion
    
    mejor_error_actual <- metricas[mejor_iteracion_actual]
    

    
    if(mejor_error_actual < mejor_error){
      
      mejor_error <- mejor_error_actual
      
      mejores_parametros <- parametros_actuales
      
      mejor_iteracion <- mejor_iteracion_actual
      
      mejor_num_iterations <- num_iterations_actual
      
    }
    
  }
  
  
  # Modelo final
  
  lgbm_final <- lgb.train(
    params = mejores_parametros,
    data = datos_lgbm,
    nrounds = mejor_iteracion,
    verbose = -1
  )
  
  
  # Guardamos parametros para reportar
  
  mejores_parametros_salida <- c(
    list(
      num_iterations = mejor_num_iterations
    ),
    mejores_parametros,
    list(
      best_iteration = mejor_iteracion
    )
  )
  
  
  # Importancia de variables
  
  importancia <- lgb.importance(
    model = lgbm_final,
    percentage = TRUE
  )
  
  importancia <- head(
    importancia[order(-importancia$Gain), ],
    10
  )
  
  
  # Medidas aparentes
  
  predicciones <- predict(
    lgbm_final,
    X
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
  
  
  tiempo <- toc(quiet = TRUE)
  
  
  return(list(
    nombreModelo = nombre_modelo,
    metricasAparentes = respond,
    matrizConfusion = matriz$table,
    importancia = importancia,
    modelo = lgbm_final,
    mejores_parametros = mejores_parametros_salida,
    tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
  
}


# Cálculo del poder predictivo -------------------------------------------------

LGBM.PP <- function(folds, nombre_modelo, genesKP = "mezcla"){
  
  tic()
  set.seed(123)
  
  modLGMB <- function(split){
    
    # Train y Test
    
    train_data <- analysis(split)
    test_data  <- assessment(split)
    
    
    # Filtramos por las variables de interes
    
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
    
    
    # Dataset LightGBM
    
    lgbm_train <- lgb.Dataset(
      data = X_train,
      label = Y_train
    )
    
    
    # Malla de hiperparametros
    
    malla <- expand.grid(
      num_iterations = c(250, 500),
      learning_rate = c(0.01, 0.1),
      num_leaves = c(4, 8, 16),
      top_rate = c(0.2, 0.3),
      other_rate = c(0.1, 0.2),
      min_data_in_leaf = c(10, 20)
    )
    
    
    # Parametros base
    
    parametros_base <- list(
      objective = "binary",
      metric = "binary_logloss",
      data_sample_strategy = "goss",
      verbosity = -1,
      feature_pre_filter = FALSE,
      seed = 123
    )
    
    
    mejor_error <- Inf
    mejores_parametros <- NULL
    mejor_iteracion <- NULL
    
    
    # Tuneo de hiperparametros
    
    for (i in 1:nrow(malla)) {
      
      # Numero maximo de iteraciones de esta combinacion
      
      num_iterations_actual <- malla$num_iterations[i]
      
      
      # Construimos parametros sin num_iterations,
      # ya que este se pasa mediante nrounds
      
      parametros_actuales <- c(
        parametros_base,
        as.list(malla[i,setdiff(names(malla), "num_iterations")])
      )
      
      
      # Validacion cruzada interna
      
      lgbm_tune <- lgb.cv(
        params = parametros_actuales,
        data = lgbm_train,
        nrounds = num_iterations_actual,
        nfold = 5,
        early_stopping_rounds = 10,
        verbose = -1
      )
      
      
      # Metricas de validacion
      
      metricas <- unlist(
        lgbm_tune$record_evals$valid$binary_logloss$eval
      )
      
      
      # Mejor iteracion encontrada por early stopping
      
      if(!is.null(lgbm_tune$best_iter) &&
         lgbm_tune$best_iter > 0){
        
        mejor_iteracion_actual <- lgbm_tune$best_iter
        
      } else {
        
        # Si no se activa early stopping
        mejor_iteracion_actual <- which.min(metricas)
        
      }
      
      
      # Error correspondiente a la mejor iteracion
      
      mejor_error_actual <- metricas[mejor_iteracion_actual]
      
      
      # Guardamos la mejor combinacion
      
      if(mejor_error_actual < mejor_error){
        
        mejor_error <- mejor_error_actual
        mejores_parametros <- parametros_actuales
        mejor_iteracion <- mejor_iteracion_actual
        
      }
      
    }
    
    
    # Modelo final del fold
    
    lgbm_final <- lgb.train(
      params = mejores_parametros,
      data = lgbm_train,
      nrounds = mejor_iteracion,
      verbose = -1
    )
    
    
    # Predicciones sobre conjunto de prueba
    
    predicciones <- predict(
      lgbm_final,
      X_test
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
    modLGMB
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
  
  
  tiempo <- toc(quiet = TRUE)
  
  
  # Resultados
  
  return(list(
    nombreModelo = nombre_modelo,
    resultados = resultados,
    tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
  
}

# LightGBM ---------------------------------------------------------------------

# GRUPO 1 ----------------------------------------------------------------------

# BRAIN

M6.LGBM.Brain.Grupo1.Final <- LGBM(
  df = datos_brain, 
  nombre_modelo = "M6 LGBM Brain Grupo1 Final", 
  genesKP = "grupo_1"
)

M6.LGBM.Brain.Grupo1.PP <- LGBM.PP(
  folds = folds_brain, 
  nombre_modelo = "M6 LGBM Brain Grupo1 PP", 
  genesKP = "grupo_1"
)


# SKIN

M6.LGBM.Skin.Grupo1.Final <- LGBM(
  df = datos_skin, 
  nombre_modelo = "M6 LGBM Skin Grupo1 Final", 
  genesKP = "grupo_1"
)

M6.LGBM.Skin.Grupo1.PP <- LGBM.PP(
  folds = folds_skin, 
  nombre_modelo = "M6 LGBM Skin Grupo1 PP", 
  genesKP = "grupo_1"
)


# LUNG

M6.LGBM.Lung.Grupo1.Final <- LGBM(
  df = datos_lung, 
  nombre_modelo = "M6 LGBM Lung Grupo1 Final", 
  genesKP = "grupo_1"
)

M6.LGBM.Lung.Grupo1.PP <- LGBM.PP(
  folds = folds_lung, 
  nombre_modelo = "M6 LGBM Lung Grupo1 PP", 
  genesKP = "grupo_1"
)


# STOMACH

M6.LGBM.Stomach.Grupo1.Final <- LGBM(
  df = datos_stomach, 
  nombre_modelo = "M6 LGBM Stomach Grupo1 Final", 
  genesKP = "grupo_1"
)

M6.LGBM.Stomach.Grupo1.PP <- LGBM.PP(
  folds = folds_stomach, 
  nombre_modelo = "M6 LGBM Stomach Grupo1 PP", 
  genesKP = "grupo_1"
)


# THYROID

M6.LGBM.Thyroid.Grupo1.Final <- LGBM(
  df = datos_thyroid, 
  nombre_modelo = "M6 LGBM Thyroid Grupo1 Final", 
  genesKP = "grupo_1"
)

M6.LGBM.Thyroid.Grupo1.PP <- LGBM.PP(
  folds = folds_thyroid, 
  nombre_modelo = "M6 LGBM Thyroid Grupo1 PP", 
  genesKP = "grupo_1"
)


# BREAST

M6.LGBM.Breast.Grupo1.Final <- LGBM(
  df = datos_breast, 
  nombre_modelo = "M6 LGBM Breast Grupo1 Final", 
  genesKP = "grupo_1"
)

M6.LGBM.Breast.Grupo1.PP <- LGBM.PP(
  folds = folds_breast, 
  nombre_modelo = "M6 LGBM Breast Grupo1 PP", 
  genesKP = "grupo_1"
)


# Guardamos resultados

resultados_lgbm_grupo1 <- list(
  
  # BRAIN
  
  M6.LGBM.Brain.Grupo1.Final = M6.LGBM.Brain.Grupo1.Final,
  M6.LGBM.Brain.Grupo1.PP    = M6.LGBM.Brain.Grupo1.PP,
  
  
  # SKIN
  
  M6.LGBM.Skin.Grupo1.Final = M6.LGBM.Skin.Grupo1.Final,
  M6.LGBM.Skin.Grupo1.PP    = M6.LGBM.Skin.Grupo1.PP,
  
  
  # LUNG
  
  M6.LGBM.Lung.Grupo1.Final = M6.LGBM.Lung.Grupo1.Final,
  M6.LGBM.Lung.Grupo1.PP    = M6.LGBM.Lung.Grupo1.PP,
  
  
  # STOMACH
  
  M6.LGBM.Stomach.Grupo1.Final = M6.LGBM.Stomach.Grupo1.Final,
  M6.LGBM.Stomach.Grupo1.PP    = M6.LGBM.Stomach.Grupo1.PP,
  
  
  # THYROID
  
  M6.LGBM.Thyroid.Grupo1.Final = M6.LGBM.Thyroid.Grupo1.Final,
  M6.LGBM.Thyroid.Grupo1.PP    = M6.LGBM.Thyroid.Grupo1.PP,
  
  
  # BREAST
  
  M6.LGBM.Breast.Grupo1.Final = M6.LGBM.Breast.Grupo1.Final,
  M6.LGBM.Breast.Grupo1.PP    = M6.LGBM.Breast.Grupo1.PP
)


save(
  resultados_lgbm_grupo1, 
  file = "resultados/resultados_LGBM_grupo1.RData"
)



# GRUPO 2 ----------------------------------------------------------------------

# BRAIN

M6.LGBM.Brain.Grupo2.Final <- LGBM(
  df = datos_brain, 
  nombre_modelo = "M6 LGBM Brain Grupo2 Final", 
  genesKP = "grupo_2"
)

M6.LGBM.Brain.Grupo2.PP <- LGBM.PP(
  folds = folds_brain, 
  nombre_modelo = "M6 LGBM Brain Grupo2 PP", 
  genesKP = "grupo_2"
)


# SKIN

M6.LGBM.Skin.Grupo2.Final <- LGBM(
  df = datos_skin, 
  nombre_modelo = "M6 LGBM Skin Grupo2 Final", 
  genesKP = "grupo_2"
)

M6.LGBM.Skin.Grupo2.PP <- LGBM.PP(
  folds = folds_skin, 
  nombre_modelo = "M6 LGBM Skin Grupo2 PP", 
  genesKP = "grupo_2"
)


# LUNG

M6.LGBM.Lung.Grupo2.Final <- LGBM(
  df = datos_lung, 
  nombre_modelo = "M6 LGBM Lung Grupo2 Final", 
  genesKP = "grupo_2"
)

M6.LGBM.Lung.Grupo2.PP <- LGBM.PP(
  folds = folds_lung, 
  nombre_modelo = "M6 LGBM Lung Grupo2 PP", 
  genesKP = "grupo_2"
)


# STOMACH

M6.LGBM.Stomach.Grupo2.Final <- LGBM(
  df = datos_stomach, 
  nombre_modelo = "M6 LGBM Stomach Grupo2 Final", 
  genesKP = "grupo_2"
)

M6.LGBM.Stomach.Grupo2.PP <- LGBM.PP(
  folds = folds_stomach, 
  nombre_modelo = "M6 LGBM Stomach Grupo2 PP", 
  genesKP = "grupo_2"
)


# THYROID

M6.LGBM.Thyroid.Grupo2.Final <- LGBM(
  df = datos_thyroid, 
  nombre_modelo = "M6 LGBM Thyroid Grupo2 Final", 
  genesKP = "grupo_2"
)

M6.LGBM.Thyroid.Grupo2.PP <- LGBM.PP(
  folds = folds_thyroid, 
  nombre_modelo = "M6 LGBM Thyroid Grupo2 PP", 
  genesKP = "grupo_2"
)


# BREAST

M6.LGBM.Breast.Grupo2.Final <- LGBM(
  df = datos_breast, 
  nombre_modelo = "M6 LGBM Breast Grupo2 Final", 
  genesKP = "grupo_2"
)

M6.LGBM.Breast.Grupo2.PP <- LGBM.PP(
  folds = folds_breast, 
  nombre_modelo = "M6 LGBM Breast Grupo2 PP", 
  genesKP = "grupo_2"
)


# Guardamos resultados

resultados_lgbm_grupo2 <- list(
  
  # BRAIN
  
  M6.LGBM.Brain.Grupo2.Final = M6.LGBM.Brain.Grupo2.Final,
  M6.LGBM.Brain.Grupo2.PP    = M6.LGBM.Brain.Grupo2.PP,
  
  
  # SKIN
  
  M6.LGBM.Skin.Grupo2.Final = M6.LGBM.Skin.Grupo2.Final,
  M6.LGBM.Skin.Grupo2.PP    = M6.LGBM.Skin.Grupo2.PP,
  
  
  # LUNG
  
  M6.LGBM.Lung.Grupo2.Final = M6.LGBM.Lung.Grupo2.Final,
  M6.LGBM.Lung.Grupo2.PP    = M6.LGBM.Lung.Grupo2.PP,
  
  
  # STOMACH
  
  M6.LGBM.Stomach.Grupo2.Final = M6.LGBM.Stomach.Grupo2.Final,
  M6.LGBM.Stomach.Grupo2.PP    = M6.LGBM.Stomach.Grupo2.PP,
  
  
  # THYROID
  
  M6.LGBM.Thyroid.Grupo2.Final = M6.LGBM.Thyroid.Grupo2.Final,
  M6.LGBM.Thyroid.Grupo2.PP    = M6.LGBM.Thyroid.Grupo2.PP,
  
  
  # BREAST
  
  M6.LGBM.Breast.Grupo2.Final = M6.LGBM.Breast.Grupo2.Final,
  M6.LGBM.Breast.Grupo2.PP    = M6.LGBM.Breast.Grupo2.PP
)


save(
  resultados_lgbm_grupo2, 
  file = "resultados/resultados_LGBM_grupo2.RData"
)


# GRUPO 3 ----------------------------------------------------------------------

# BRAIN

M6.LGBM.Brain.Grupo3.Final <- LGBM(
  df = datos_brain, 
  nombre_modelo = "M6 LGBM Brain Grupo3 Final", 
  genesKP = "grupo_3"
)

M6.LGBM.Brain.Grupo3.PP <- LGBM.PP(
  folds = folds_brain, 
  nombre_modelo = "M6 LGBM Brain Grupo3 PP", 
  genesKP = "grupo_3"
)


# SKIN

M6.LGBM.Skin.Grupo3.Final <- LGBM(
  df = datos_skin, 
  nombre_modelo = "M6 LGBM Skin Grupo3 Final", 
  genesKP = "grupo_3"
)

M6.LGBM.Skin.Grupo3.PP <- LGBM.PP(
  folds = folds_skin, 
  nombre_modelo = "M6 LGBM Skin Grupo3 PP", 
  genesKP = "grupo_3"
)


# LUNG

M6.LGBM.Lung.Grupo3.Final <- LGBM(
  df = datos_lung, 
  nombre_modelo = "M6 LGBM Lung Grupo3 Final", 
  genesKP = "grupo_3"
)

M6.LGBM.Lung.Grupo3.PP <- LGBM.PP(
  folds = folds_lung, 
  nombre_modelo = "M6 LGBM Lung Grupo3 PP", 
  genesKP = "grupo_3"
)


# STOMACH

M6.LGBM.Stomach.Grupo3.Final <- LGBM(
  df = datos_stomach, 
  nombre_modelo = "M6 LGBM Stomach Grupo3 Final", 
  genesKP = "grupo_3"
)

M6.LGBM.Stomach.Grupo3.PP <- LGBM.PP(
  folds = folds_stomach, 
  nombre_modelo = "M6 LGBM Stomach Grupo3 PP", 
  genesKP = "grupo_3"
)


# THYROID

M6.LGBM.Thyroid.Grupo3.Final <- LGBM(
  df = datos_thyroid, 
  nombre_modelo = "M6 LGBM Thyroid Grupo3 Final", 
  genesKP = "grupo_3"
)

M6.LGBM.Thyroid.Grupo3.PP <- LGBM.PP(
  folds = folds_thyroid, 
  nombre_modelo = "M6 LGBM Thyroid Grupo3 PP", 
  genesKP = "grupo_3"
)


# BREAST

M6.LGBM.Breast.Grupo3.Final <- LGBM(
  df = datos_breast, 
  nombre_modelo = "M6 LGBM Breast Grupo3 Final", 
  genesKP = "grupo_3"
)

M6.LGBM.Breast.Grupo3.PP <- LGBM.PP(
  folds = folds_breast, 
  nombre_modelo = "M6 LGBM Breast Grupo3 PP", 
  genesKP = "grupo_3"
)


# Guardamos resultados

resultados_lgbm_grupo3 <- list(
  
  # BRAIN
  
  M6.LGBM.Brain.Grupo3.Final = M6.LGBM.Brain.Grupo3.Final,
  M6.LGBM.Brain.Grupo3.PP    = M6.LGBM.Brain.Grupo3.PP,
  
  
  # SKIN
  
  M6.LGBM.Skin.Grupo3.Final = M6.LGBM.Skin.Grupo3.Final,
  M6.LGBM.Skin.Grupo3.PP    = M6.LGBM.Skin.Grupo3.PP,
  
  
  # LUNG
  
  M6.LGBM.Lung.Grupo3.Final = M6.LGBM.Lung.Grupo3.Final,
  M6.LGBM.Lung.Grupo3.PP    = M6.LGBM.Lung.Grupo3.PP,
  
  
  # STOMACH
  
  M6.LGBM.Stomach.Grupo3.Final = M6.LGBM.Stomach.Grupo3.Final,
  M6.LGBM.Stomach.Grupo3.PP    = M6.LGBM.Stomach.Grupo3.PP,
  
  
  # THYROID
  
  M6.LGBM.Thyroid.Grupo3.Final = M6.LGBM.Thyroid.Grupo3.Final,
  M6.LGBM.Thyroid.Grupo3.PP    = M6.LGBM.Thyroid.Grupo3.PP,
  
  
  # BREAST
  
  M6.LGBM.Breast.Grupo3.Final = M6.LGBM.Breast.Grupo3.Final,
  M6.LGBM.Breast.Grupo3.PP    = M6.LGBM.Breast.Grupo3.PP
)


save(
  resultados_lgbm_grupo3, 
  file = "resultados/resultados_LGBM_grupo3.RData"
)

























