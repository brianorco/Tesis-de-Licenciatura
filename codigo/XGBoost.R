# XGBoost (Extreme Gradient Boosting) ------------------------------------------


# Paqueterias ------------------------------------------------------------------

#install.packages("xgboost")
library(xgboost)
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

# XGBoost ----------------------------------------------------------------------

# Modelo Final -----------------------------------------------------------------


XGB <- function(df, nombre_modelo, genesKP = "mezcla"){
  
  tic()
  set.seed(123)
  
  #Filtramos variables de interes
  
  if(genesKP == "grupo_1"){
    df_filtrado <- df[ , c(grupo_1, 'clase')]
  } else if (genesKP == "grupo_2"){
    df_filtrado <- df[ , c(grupo_2, 'clase')]
  } else if (genesKP == "grupo_3"){
    df_filtrado <- df[ , c(grupo_3, 'clase')]
  } else if (genesKP == "mezcla") {
    df_filtrado <- df
  }
  
  # Tuneamos los hiperparametros
  
  malla <- expand.grid(
    nrounds = c(250,500) #Número de arboles 
    , max_depth = c(2,3,4) #Profundidad máxima de los árboles
    , eta = c( .01, .1 ) #Tamaño del paso 
    , gamma = c(.01 ,1) #Reducción mínima de pérdida para un split 
    , colsample_bytree = c(.5,.8) #Fracción de variables por árbol
    , min_child_weight = c(1, 3) #Peso mínimo en nodos hoja
    , subsample = c(.8, 1) #Fracción de muestras por árbol
   )
  
  ctrl <- trainControl(
    method = "cv"
    , number = 5
    , classProbs = TRUE
    , summaryFunction = twoClassSummary
    , allowParallel = TRUE
  )
  
  xgb_tune <- train(
    x = df_filtrado[,-which(colnames(df_filtrado) == "clase")]
    , y = df_filtrado$clase
    , method = "xgbTree"
    , trControl = ctrl
    , tuneGrid = malla
    , metric = "Sens"
    , verbosity = 0
  )
  
  mejores_parametros <- xgb_tune$bestTune
  
  parametros <- list(
    objective = "binary:logistic"
    , max_depth = mejores_parametros$max_depth
    , eta = mejores_parametros$eta
    , gamma = mejores_parametros$gamma
    , colsample_bytree = mejores_parametros$colsample_bytree
    , min_child_weight = mejores_parametros$min_child_weight
    , subsample = mejores_parametros$subsample
    , eval_metric = "logloss"
  )
  
  # Preparamos los datos para XGBoost
  
  x_datos <- as.matrix(df_filtrado[,-which(colnames(df_filtrado) == "clase")])
  y_datos <- as.numeric(df_filtrado$clase) - 1 
  
  matriz_xgb <- xgb.DMatrix(data = x_datos, label = y_datos)
  
  xgb_final <- xgb.train(
    params = parametros
    , data = matriz_xgb
    , nrounds = mejores_parametros$nrounds
    , verbose = 0
  ) 
  
  importancia <- xgb.importance(model = xgb_final)
  importancia <- head(importancia[order(-importancia$Gain),],10)
  
  #Medidas Aparentes
  predicciones <- predict(xgb_final, newdata = matriz_xgb)
  predicciones_clase <- ifelse(predicciones >= 0.5, 
                               levels(df_filtrado$clase)[2], 
                               levels(df_filtrado$clase)[1])
  
  matriz <- confusionMatrix(
    data = factor(predicciones_clase, levels = levels(df_filtrado$clase))
    , reference = df_filtrado$clase
    , positive = levels(df_filtrado$clase)[2]
  )
  
  respond <- data.frame(
    Metric = c("accuracy", "recall", "specificity")
    , Score = c(
      as.numeric(matriz$overall["Accuracy"])
      , as.numeric(matriz$byClass["Sensitivity"])
      , as.numeric(matriz$byClass["Specificity"])
    )
  )
  
  tiempo <- toc(quiet = TRUE)
  
  return(list(
    nombreModelo = nombre_modelo
    , metricasAparentes = respond
    , matrizConfusion = matriz$table
    , importancia = importancia
    , modelo = xgb_final
    , tiempoEjecucion = tiempo$toc - tiempo$tic 
  ))
}


# Cálculo del poder predictivo -------------------------------------------------

XGB.PP <- function(folds, nombre_modelo, genesKP = "mezcla"){
  tic()
  set.seed(123)
  
  XGBCV <- function(split){
    # Train y Test
    train_data <- analysis(split)
    test_data  <- assessment(split)
    
    # Filtramos por las variables de interes
    if(genesKP == "grupo_1"){
      train_data_filtrado <- train_data[ , c(grupo_1, 'clase')]
      test_data_filtrado <- test_data[ , c(grupo_1, 'clase')]
    } else if (genesKP == "grupo_2"){
      train_data_filtrado <- train_data[ , c(grupo_2, 'clase')]
      test_data_filtrado <- test_data[ , c(grupo_2, 'clase')]
    } else if (genesKP == "grupo_3"){
      train_data_filtrado <- train_data[ , c(grupo_3, 'clase')]
      test_data_filtrado <- test_data[ , c(grupo_3, 'clase')]
    } else if (genesKP == "mezcla"){
      train_data_filtrado <- train_data
      test_data_filtrado <- test_data
    }
    
    # Tuneamos los hiperparametros
    
    malla <- expand.grid(
      nrounds = c(50,100,150) #Número de arboles 
      , max_depth = c(10,20) #Profundidad máxima de los árboles
      , eta = c(.001, .01, .1 ) #Tamaño del paso 
      , gamma = c(0,1) #Reducción mínima de pérdida para un split **
      , colsample_bytree = c(.5,.8) #Fracción de variables por árbol
      , min_child_weight = c(1, 3) #Peso mínimo en nodos hoja
      , subsample = c(.8, 1) #Fracción de muestras por árbol
    )
    
    ctrl <- trainControl(
      method = "cv"
      , number = 5
      , classProbs = TRUE
      , summaryFunction = twoClassSummary
      , allowParallel = TRUE
    )
    
    xgb_tune <- train(
      x = train_data_filtrado[,-which(colnames(train_data_filtrado) == "clase")]
      , y = train_data_filtrado$clase
      , method = "xgbTree"
      , trControl = ctrl
      , tuneGrid = malla
      , metric = "Sens"
      , verbosity = 0
    )
    
    mejores_parametros <- xgb_tune$bestTune
    
    parametros <- list(
      objective = "binary:logistic"
      , max_depth = mejores_parametros$max_depth
      , eta = mejores_parametros$eta
      , gamma = mejores_parametros$gamma
      , colsample_bytree = mejores_parametros$colsample_bytree
      , min_child_weight = mejores_parametros$min_child_weight
      , subsample = mejores_parametros$subsample
      , eval_metric = "logloss"
    )
    
    # Preparamos los datos para XGBoost
    
    x_datos_train <- as.matrix(train_data_filtrado[,-which(colnames(train_data_filtrado) == "clase")])
    x_datos_test <- as.matrix(test_data_filtrado[,-which(colnames(test_data_filtrado) == "clase")])
    
    y_datos_train <- as.numeric(train_data_filtrado$clase) - 1 
    y_datos_test <- test_data_filtrado$clase
    
    matriz_xgb_train <- xgb.DMatrix(data = x_datos_train, label = y_datos_train)
    matriz_xgb_test <- xgb.DMatrix(data = x_datos_test)
    
    # Modelo Final
    xgb_final <- xgb.train(
      params = parametros
      , data = matriz_xgb_train
      , nrounds = mejores_parametros$nrounds
      , verbose = 0
    ) 
    
    predicciones <- predict(xgb_final, newdata = matriz_xgb_test)
    predicciones_clase <- ifelse(predicciones >= 0.5, 
                                 levels(train_data_filtrado$clase)[2], 
                                 levels(train_data_filtrado$clase)[1])
    
    matriz <- confusionMatrix(
      data = factor(predicciones_clase, levels = levels(train_data_filtrado$clase))
      , reference = y_datos_test
      , positive = levels(train_data_filtrado$clase)[2]
    )
    
    respond <- data.frame(
      Metric = c("accuracy", "recall", "specificity")
      , Score = c(
        as.numeric(matriz$overall["Accuracy"])
        , as.numeric(matriz$byClass["Sensitivity"])
        , as.numeric(matriz$byClass["Specificity"])
      )
    )
    
    return(respond[,2])
    
  }
  
  resultados_folds <- lapply(folds$splits, XGBCV)
  
  promedio_metricas <- rowMeans(do.call(cbind, resultados_folds))
  
  resultados <- data.frame(
    metricas = c("accuracy", "recall", "specificity")
    , promedio = promedio_metricas
  )
  
  tiempo <- toc(quiet = TRUE)
  
  return(list(
    nombreModelo = nombre_modelo
    , resultados = resultados
    , tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
}


# XGBoost ----------------------------------------------------------------------

# BRAIN -------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M5.XGB.Brain.Grupo1.Final <- XGB(df = datos_brain, nombre_modelo = "M5 XGB Brain Grupo1 Final", genesKP = "grupo_1")
M5.XGB.Brain.Grupo1.PP <- XGB.PP(folds = folds_brain, nombre_modelo = "M5 XGB Brain Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M5.XGB.Brain.Grupo2.Final <- XGB(df = datos_brain, nombre_modelo = "M5 XGB Brain Grupo2 Final", genesKP = "grupo_2")
M5.XGB.Brain.Grupo2.PP <- XGB.PP(folds = folds_brain, nombre_modelo = "M5 XGB Brain Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M5.XGB.Brain.Grupo3.Final <- XGB(df = datos_brain, nombre_modelo = "M5 XGB Brain Grupo3 Final", genesKP = "grupo_3")
M5.XGB.Brain.Grupo3.PP <- XGB.PP(folds = folds_brain, nombre_modelo = "M5 XGB Brain Grupo3 PP", genesKP = "grupo_3")

# SKIN -------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M5.XGB.Skin.Grupo1.Final <- XGB(df = datos_skin, nombre_modelo = "M5 XGB Skin Grupo1 Final", genesKP = "grupo_1")
M5.XGB.Skin.Grupo1.PP <- XGB.PP(folds = folds_skin, nombre_modelo = "M5 XGB Skin Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M5.XGB.Skin.Grupo2.Final <- XGB(df = datos_skin, nombre_modelo = "M5 XGB Skin Grupo2 Final", genesKP = "grupo_2")
M5.XGB.Skin.Grupo2.PP <- XGB.PP(folds = folds_skin, nombre_modelo = "M5 XGB Skin Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M5.XGB.Skin.Grupo3.Final <- XGB(df = datos_skin, nombre_modelo = "M5 XGB Skin Grupo3 Final", genesKP = "grupo_3")
M5.XGB.Skin.Grupo3.PP <- XGB.PP(folds = folds_skin, nombre_modelo = "M5 XGB Skin Grupo3 PP", genesKP = "grupo_3")



resultados_xgb_BrainSkin <- list(
  
  # Brain
  M5.XGB.Brain.Grupo1.Final = M5.XGB.Brain.Grupo1.Final,
  M5.XGB.Brain.Grupo1.PP = M5.XGB.Brain.Grupo1.PP,
  
  M5.XGB.Brain.Grupo2.Final = M5.XGB.Brain.Grupo2.Final,
  M5.XGB.Brain.Grupo2.PP = M5.XGB.Brain.Grupo2.PP,
  
  M5.XGB.Brain.Grupo3.Final = M5.XGB.Brain.Grupo3.Final,
  M5.XGB.Brain.Grupo3.PP = M5.XGB.Brain.Grupo3.PP,
  
  
  # Skin
  M5.XGB.Skin.Grupo1.Final = M5.XGB.Skin.Grupo1.Final,
  M5.XGB.Skin.Grupo1.PP = M5.XGB.Skin.Grupo1.PP,
  
  M5.XGB.Skin.Grupo2.Final = M5.XGB.Skin.Grupo2.Final,
  M5.XGB.Skin.Grupo2.PP = M5.XGB.Skin.Grupo2.PP,
  
  M5.XGB.Skin.Grupo3.Final = M5.XGB.Skin.Grupo3.Final,
  M5.XGB.Skin.Grupo3.PP = M5.XGB.Skin.Grupo3.PP
  
)



# Guardar la lista completa
save(resultados_xgb_BrainSkin, file = "resultados/resultados_XGB_BrainSkin.RData")



# LUNG -------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M5.XGB.Lung.Grupo1.Final <- XGB(df = datos_lung, nombre_modelo = "M5 XGB Lung Grupo1 Final", genesKP = "grupo_1")
M5.XGB.Lung.Grupo1.PP <- XGB.PP(folds = folds_lung, nombre_modelo = "M5 XGB Lung Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M5.XGB.Lung.Grupo2.Final <- XGB(df = datos_lung, nombre_modelo = "M5 XGB Lung Grupo2 Final", genesKP = "grupo_2")
M5.XGB.Lung.Grupo2.PP <- XGB.PP(folds = folds_lung, nombre_modelo = "M5 XGB Lung Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M5.XGB.Lung.Grupo3.Final <- XGB(df = datos_lung, nombre_modelo = "M5 XGB Lung Grupo3 Final", genesKP = "grupo_3")
M5.XGB.Lung.Grupo3.PP <- XGB.PP(folds = folds_lung, nombre_modelo = "M5 XGB Lung Grupo3 PP", genesKP = "grupo_3")

# STOMACH ----------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M5.XGB.Stomach.Grupo1.Final <- XGB(df = datos_stomach, nombre_modelo = "M5 XGB Stomach Grupo1 Final", genesKP = "grupo_1")
M5.XGB.Stomach.Grupo1.PP <- XGB.PP(folds = folds_stomach, nombre_modelo = "M5 XGB Stomach Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M5.XGB.Stomach.Grupo2.Final <- XGB(df = datos_stomach, nombre_modelo = "M5 XGB Stomach Grupo2 Final", genesKP = "grupo_2")
M5.XGB.Stomach.Grupo2.PP <- XGB.PP(folds = folds_stomach, nombre_modelo = "M5 XGB Stomach Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M5.XGB.Stomach.Grupo3.Final <- XGB(df = datos_stomach, nombre_modelo = "M5 XGB Stomach Grupo3 Final", genesKP = "grupo_3")
M5.XGB.Stomach.Grupo3.PP <- XGB.PP(folds = folds_stomach, nombre_modelo = "M5 XGB Stomach Grupo3 PP", genesKP = "grupo_3")



resultados_xgb_LungStomach <- list(
  
  # Lung
  M5.XGB.Lung.Grupo1.Final = M5.XGB.Lung.Grupo1.Final,
  M5.XGB.Lung.Grupo1.PP = M5.XGB.Lung.Grupo1.PP,
  
  M5.XGB.Lung.Grupo2.Final = M5.XGB.Lung.Grupo2.Final,
  M5.XGB.Lung.Grupo2.PP = M5.XGB.Lung.Grupo2.PP,
  
  M5.XGB.Lung.Grupo3.Final = M5.XGB.Lung.Grupo3.Final,
  M5.XGB.Lung.Grupo3.PP = M5.XGB.Lung.Grupo3.PP,
  
  
  # Stomach
  M5.XGB.Stomach.Grupo1.Final = M5.XGB.Stomach.Grupo1.Final,
  M5.XGB.Stomach.Grupo1.PP = M5.XGB.Stomach.Grupo1.PP,
  
  M5.XGB.Stomach.Grupo2.Final = M5.XGB.Stomach.Grupo2.Final,
  M5.XGB.Stomach.Grupo2.PP = M5.XGB.Stomach.Grupo2.PP,
  
  M5.XGB.Stomach.Grupo3.Final = M5.XGB.Stomach.Grupo3.Final,
  M5.XGB.Stomach.Grupo3.PP = M5.XGB.Stomach.Grupo3.PP
)



# Guardar la lista completa
save(resultados_xgb_LungStomach, file = "resultados/resultados_XGB_LungStomach.RData")







# THYROID ----------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M5.XGB.Thyroid.Grupo1.Final <- XGB(df = datos_thyroid, nombre_modelo = "M5 XGB Thyroid Grupo1 Final", genesKP = "grupo_1")
M5.XGB.Thyroid.Grupo1.PP <- XGB.PP(folds = folds_thyroid, nombre_modelo = "M5 XGB Thyroid Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M5.XGB.Thyroid.Grupo2.Final <- XGB(df = datos_thyroid, nombre_modelo = "M5 XGB Thyroid Grupo2 Final", genesKP = "grupo_2")
M5.XGB.Thyroid.Grupo2.PP <- XGB.PP(folds = folds_thyroid, nombre_modelo = "M5 XGB Thyroid Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M5.XGB.Thyroid.Grupo3.Final <- XGB(df = datos_thyroid, nombre_modelo = "M5 XGB Thyroid Grupo3 Final", genesKP = "grupo_3")
M5.XGB.Thyroid.Grupo3.PP <- XGB.PP(folds = folds_thyroid, nombre_modelo = "M5 XGB Thyroid Grupo3 PP", genesKP = "grupo_3")

# BREAST -----------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M5.XGB.Breast.Grupo1.Final <- XGB(df = datos_breast, nombre_modelo = "M5 XGB Breast Grupo1 Final", genesKP = "grupo_1")
M5.XGB.Breast.Grupo1.PP <- XGB.PP(folds = folds_breast, nombre_modelo = "M5 XGB Breast Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M5.XGB.Breast.Grupo2.Final <- XGB(df = datos_breast, nombre_modelo = "M5 XGB Breast Grupo2 Final", genesKP = "grupo_2")
M5.XGB.Breast.Grupo2.PP <- XGB.PP(folds = folds_breast, nombre_modelo = "M5 XGB Breast Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M5.XGB.Breast.Grupo3.Final <- XGB(df = datos_breast, nombre_modelo = "M5 XGB Breast Grupo3 Final", genesKP = "grupo_3")
M5.XGB.Breast.Grupo3.PP <- XGB.PP(folds = folds_breast, nombre_modelo = "M5 XGB Breast Grupo3 PP", genesKP = "grupo_3")



resultados_xgb_ThyroidBreast <- list(
  
  # Thyroid
  M5.XGB.Thyroid.Grupo1.Final = M5.XGB.Thyroid.Grupo1.Final,
  M5.XGB.Thyroid.Grupo1.PP = M5.XGB.Thyroid.Grupo1.PP,
  
  M5.XGB.Thyroid.Grupo2.Final = M5.XGB.Thyroid.Grupo2.Final,
  M5.XGB.Thyroid.Grupo2.PP = M5.XGB.Thyroid.Grupo2.PP,
  
  M5.XGB.Thyroid.Grupo3.Final = M5.XGB.Thyroid.Grupo3.Final,
  M5.XGB.Thyroid.Grupo3.PP = M5.XGB.Thyroid.Grupo3.PP,
  
  
  # Breast
  M5.XGB.Breast.Grupo1.Final = M5.XGB.Breast.Grupo1.Final,
  M5.XGB.Breast.Grupo1.PP = M5.XGB.Breast.Grupo1.PP,
  
  M5.XGB.Breast.Grupo2.Final = M5.XGB.Breast.Grupo2.Final,
  M5.XGB.Breast.Grupo2.PP = M5.XGB.Breast.Grupo2.PP,
  
  M5.XGB.Breast.Grupo3.Final = M5.XGB.Breast.Grupo3.Final,
  M5.XGB.Breast.Grupo3.PP = M5.XGB.Breast.Grupo3.PP
)



# Guardar la lista completa
save(resultados_xgb_ThyroidBreast, file = "resultados/resultados_XGB_ThyroidBreast.RData")










































