# Random Forest ----------------------------------------------------------------


# Paqueterias ------------------------------------------------------------------

library(plyr)
library(dplyr)
library(tidyr)
library(forcats)
library(e1071)
library(rsample)
library(caret)
library(randomForest)
library(tictoc)


# Funciones Auxiliares ---------------------------------------------------------

source("codigo/FuncionesAuxiliares.R")


# Conjuntos Train y Test -------------------------------------------------------

load("datos/Folds.RData")

# Random Forest ----------------------------------------------------------------

# Modelo final -----------------------------------------------------------------

RF <- function(df, nombre_modelo, genesKP = "mezcla"){
  tic()
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
  
  set.seed(123)
  
  rf_tune <- tune.randomForest(clase ~ .
                               , data = df_filtrado
                               , importance = FALSE
                               , mtry = seq(1,10, 2)
                               , ntree = c(50,100,200)
                               , nodesize = c(10, 20,  30) 
                               , tunecontrol = tune.control(sampling = "cross", cross = 5, nrepeat = 3)
                               )
  
  rf <- randomForest(clase ~ .
                     , data = df_filtrado
                     , importance = TRUE
                     , mtry = rf_tune$best.parameters$mtry
                     , ntree = rf_tune$best.parameters$ntree
                     , nodesize = rf_tune$best.parameters$nodesize
                     )
  
  importancia <- importance(rf)[order(-importance(rf)[, "MeanDecreaseAccuracy"]), ]
  
  importancia <- data.frame(
    Variable = rownames(importancia)
    , MeanDecreaseAccuracy = importancia[, "MeanDecreaseAccuracy"]
    , row.names = NULL
  )
  
  # Medidas aparentes
  
  predicciones <- predict(rf, newdata =df_filtrado, type = "class")
  
  matriz <- confusionMatrix(data = factor(predicciones, levels = levels(df_filtrado$clase))
                            , reference = df_filtrado$clase
                            , positive = levels(df_filtrado$clase)[2])
  
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
    , modelo = rf
    , tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
  
}

# Cálculo del porder predictivo ------------------------------------------------

RF.PP <- function(folds, nombre_modelo, genesKP = "mezcla"){
  tic()
  set.seed(123)
  modRF <- function(split){
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
    
    rf_tune <- tune.randomForest(clase ~ .
                                 , data = train_data_filtrado
                                 , importance = FALSE
                                 , mtry = seq(1,10, 2)
                                 , ntree = c(50,100,200)
                                 , nodesize = c(10,  20,  30) 
                                 , tunecontrol = tune.control(sampling = "cross", cross = 5, nrepeat = 3)
    )
    
    rf <- randomForest(clase ~ .
                       , data = train_data_filtrado
                       , importance = FALSE
                       , mtry = rf_tune$best.parameters$mtry
                       , ntree = rf_tune$best.parameters$ntree
                       , nodesize = rf_tune$best.parameters$nodesize
    )
    
    predicciones <- predict(rf, newdata = test_data_filtrado, type = "class")
    
    
    matriz <- confusionMatrix(
      data = factor(predicciones, levels = levels(train_data_filtrado$clase))
      , reference = test_data_filtrado$clase
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
  
  resultados_folds <- lapply(folds$splits, modRF)
  
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



# Random Forest ----------------------------------------------------------------

# BRAIN ------------------------------------------------------------------------
## Grupo 1 ---------------------------------------------------------------------
M4.RF.Brain.Grupo1.Final <- RF(df = datos_brain, nombre_modelo = "RF Brain Grupo1 Final", genesKP = "grupo_1")
M4.RF.Brain.Grupo1.PP <- RF.PP(folds = folds_brain, nombre_modelo = "RF Brain Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------
M4.RF.Brain.Grupo2.Final <- RF(df = datos_brain, nombre_modelo = "RF Brain Grupo2 Final", genesKP = "grupo_2")
M4.RF.Brain.Grupo2.PP <- RF.PP(folds = folds_brain, nombre_modelo = "RF Brain Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M4.RF.Brain.Grupo3.Final <- RF(df = datos_brain, nombre_modelo = "RF Brain Grupo3 Final", genesKP = "grupo_3")
M4.RF.Brain.Grupo3.PP <- RF.PP(folds = folds_brain, nombre_modelo = "RF Brain Grupo3 PP", genesKP = "grupo_3")


# SKIN -------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------
M4.RF.Skin.Grupo1.Final <- RF(df = datos_skin, nombre_modelo = "RF Skin Grupo1 Final", genesKP = "grupo_1")
M4.RF.Skin.Grupo1.PP <- RF.PP(folds = folds_skin, nombre_modelo = "RF Skin Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M4.RF.Skin.Grupo2.Final <- RF(df = datos_skin, nombre_modelo = "RF Skin Grupo2 Final", genesKP = "grupo_2")
M4.RF.Skin.Grupo2.PP <- RF.PP(folds = folds_skin, nombre_modelo = "RF Skin Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------
M4.RF.Skin.Grupo3.Final <- RF(df = datos_skin, nombre_modelo = "RF Skin Grupo3 Final", genesKP = "grupo_3")
M4.RF.Skin.Grupo3.PP <- RF.PP(folds = folds_skin, nombre_modelo = "RF Skin Grupo3 PP", genesKP = "grupo_3")


# LUNG -------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------
M4.RF.Lung.Grupo1.Final <- RF(df = datos_lung, nombre_modelo = "RF Lung Grupo1 Final", genesKP = "grupo_1")
M4.RF.Lung.Grupo1.PP <- RF.PP(folds = folds_lung, nombre_modelo = "RF Lung Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------
M4.RF.Lung.Grupo2.Final <- RF(df = datos_lung, nombre_modelo = "RF Lung Grupo2 Final", genesKP = "grupo_2")
M4.RF.Lung.Grupo2.PP <- RF.PP(folds = folds_lung, nombre_modelo = "RF Lung Grupo2 PP", genesKP = "grupo_2")
## Grupo 3 ---------------------------------------------------------------------
M4.RF.Lung.Grupo3.Final <- RF(df = datos_lung, nombre_modelo = "RF Lung Grupo3 Final", genesKP = "grupo_3")
M4.RF.Lung.Grupo3.PP <- RF.PP(folds = folds_lung, nombre_modelo = "RF Lung Grupo3 PP", genesKP = "grupo_3")




# STOMACH ----------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M4.RF.Stomach.Grupo1.Final <- RF(df = datos_stomach, nombre_modelo = "RF Stomach Grupo1 Final", genesKP = "grupo_1")
M4.RF.Stomach.Grupo1.PP <- RF.PP(folds = folds_stomach, nombre_modelo = "RF Stomach Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------
M4.RF.Stomach.Grupo2.Final <- RF(df = datos_stomach, nombre_modelo = "RF Stomach Grupo2 Final", genesKP = "grupo_2")
M4.RF.Stomach.Grupo2.PP <- RF.PP(folds = folds_stomach, nombre_modelo = "RF Stomach Grupo2 PP", genesKP = "grupo_2")
## Grupo 2 ---------------------------------------------------------------------
M4.RF.Stomach.Grupo3.Final <- RF(df = datos_stomach, nombre_modelo = "RF Stomach Grupo3 Final", genesKP = "grupo_3")
M4.RF.Stomach.Grupo3.PP <- RF.PP(folds = folds_stomach, nombre_modelo = "RF Stomach Grupo3 PP", genesKP = "grupo_3")

# THYROID ----------------------------------------------------------------------
## Grupo 1 ---------------------------------------------------------------------
M4.RF.Thyroid.Grupo1.Final <- RF(df = datos_thyroid, nombre_modelo = "RF Thyroid Grupo1 Final", genesKP = "grupo_1")
M4.RF.Thyroid.Grupo1.PP <- RF.PP(folds = folds_thyroid, nombre_modelo = "RF Thyroid Grupo1 PP", genesKP = "grupo_1")
## Grupo 2 ---------------------------------------------------------------------
M4.RF.Thyroid.Grupo2.Final <- RF(df = datos_thyroid, nombre_modelo = "RF Thyroid Grupo2 Final", genesKP = "grupo_2")
M4.RF.Thyroid.Grupo2.PP <- RF.PP(folds = folds_thyroid, nombre_modelo = "RF Thyroid Grupo2 PP", genesKP = "grupo_2")
## Grupo 3 ---------------------------------------------------------------------
M4.RF.Thyroid.Grupo3.Final <- RF(df = datos_thyroid, nombre_modelo = "RF Thyroid Grupo3 Final", genesKP = "grupo_3")
M4.RF.Thyroid.Grupo3.PP <- RF.PP(folds = folds_thyroid, nombre_modelo = "RF Thyroid Grupo3 PP", genesKP = "grupo_3")

# BREAST -----------------------------------------------------------------------
## Grupo 1 ---------------------------------------------------------------------
M4.RF.Breast.Grupo1.Final <- RF(df = datos_breast, nombre_modelo = "RF Breast Grupo1 Final", genesKP = "grupo_1")
M4.RF.Breast.Grupo1.PP <- RF.PP(folds = folds_breast, nombre_modelo = "RF Breast Grupo1 PP", genesKP = "grupo_1")
## Grupo 2 ---------------------------------------------------------------------
M4.RF.Breast.Grupo2.Final <- RF(df = datos_breast, nombre_modelo = "RF Breast Grupo2 Final", genesKP = "grupo_2")
M4.RF.Breast.Grupo2.PP <- RF.PP(folds = folds_breast, nombre_modelo = "RF Breast Grupo2 PP", genesKP = "grupo_2")
## Grupo 3 ---------------------------------------------------------------------
M4.RF.Breast.Grupo3.Final <- RF(df = datos_breast, nombre_modelo = "RF Breast Grupo3 Final", genesKP = "grupo_3")
M4.RF.Breast.Grupo3.PP <- RF.PP(folds = folds_breast, nombre_modelo = "RF Breast Grupo3 PP", genesKP = "grupo_3")


resultados_rf <- list(
  
  # Brain
  M4.RF.Brain.Grupo1.Final = M4.RF.Brain.Grupo1.Final,
  M4.RF.Brain.Grupo1.PP = M4.RF.Brain.Grupo1.PP,
  
  M4.RF.Brain.Grupo2.Final = M4.RF.Brain.Grupo2.Final,
  M4.RF.Brain.Grupo2.PP = M4.RF.Brain.Grupo2.PP,
  
  M4.RF.Brain.Grupo3.Final = M4.RF.Brain.Grupo3.Final,
  M4.RF.Brain.Grupo3.PP = M4.RF.Brain.Grupo3.PP,
  
  
  # Skin
  M4.RF.Skin.Grupo1.Final = M4.RF.Skin.Grupo1.Final,
  M4.RF.Skin.Grupo1.PP = M4.RF.Skin.Grupo1.PP,
  
  M4.RF.Skin.Grupo2.Final = M4.RF.Skin.Grupo2.Final,
  M4.RF.Skin.Grupo2.PP = M4.RF.Skin.Grupo2.PP,
  
  M4.RF.Skin.Grupo3.Final = M4.RF.Skin.Grupo3.Final,
  M4.RF.Skin.Grupo3.PP = M4.RF.Skin.Grupo3.PP,
  
  
  # Lung
  M4.RF.Lung.Grupo1.Final = M4.RF.Lung.Grupo1.Final,
  M4.RF.Lung.Grupo1.PP = M4.RF.Lung.Grupo1.PP,
  
  M4.RF.Lung.Grupo2.Final = M4.RF.Lung.Grupo2.Final,
  M4.RF.Lung.Grupo2.PP = M4.RF.Lung.Grupo2.PP,
  
  M4.RF.Lung.Grupo3.Final = M4.RF.Lung.Grupo3.Final,
  M4.RF.Lung.Grupo3.PP = M4.RF.Lung.Grupo3.PP,
  
  
  # Stomach
  M4.RF.Stomach.Grupo1.Final = M4.RF.Stomach.Grupo1.Final,
  M4.RF.Stomach.Grupo1.PP = M4.RF.Stomach.Grupo1.PP,
  
  M4.RF.Stomach.Grupo2.Final = M4.RF.Stomach.Grupo2.Final,
  M4.RF.Stomach.Grupo2.PP = M4.RF.Stomach.Grupo2.PP,
  
  M4.RF.Stomach.Grupo3.Final = M4.RF.Stomach.Grupo3.Final,
  M4.RF.Stomach.Grupo3.PP = M4.RF.Stomach.Grupo3.PP,
  
  
  # Thyroid
  M4.RF.Thyroid.Grupo1.Final = M4.RF.Thyroid.Grupo1.Final,
  M4.RF.Thyroid.Grupo1.PP = M4.RF.Thyroid.Grupo1.PP,
  
  M4.RF.Thyroid.Grupo2.Final = M4.RF.Thyroid.Grupo2.Final,
  M4.RF.Thyroid.Grupo2.PP = M4.RF.Thyroid.Grupo2.PP,
  
  M4.RF.Thyroid.Grupo3.Final = M4.RF.Thyroid.Grupo3.Final,
  M4.RF.Thyroid.Grupo3.PP = M4.RF.Thyroid.Grupo3.PP,
  
  
  # Breast
  M4.RF.Breast.Grupo1.Final = M4.RF.Breast.Grupo1.Final,
  M4.RF.Breast.Grupo1.PP = M4.RF.Breast.Grupo1.PP,
  
  M4.RF.Breast.Grupo2.Final = M4.RF.Breast.Grupo2.Final,
  M4.RF.Breast.Grupo2.PP = M4.RF.Breast.Grupo2.PP,
  
  M4.RF.Breast.Grupo3.Final = M4.RF.Breast.Grupo3.Final,
  M4.RF.Breast.Grupo3.PP = M4.RF.Breast.Grupo3.PP
)



save(resultados_rf, file = "resultados/resultados_RF.RData")


