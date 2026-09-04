# Classification and Regression Trees (CART) -----------------------------------


# Paqueterias ------------------------------------------------------------------

library(plyr)
library(dplyr)
library(tidyr)
library(forcats)
library(e1071)
library(rsample)
library(caret)

library(rpart) 

library(tictoc)



# Funciones Auxiliares ---------------------------------------------------------

source("codigo/FuncionesAuxiliares.R")


# Conjuntos Train y Test -------------------------------------------------------

load("datos/Folds.RData")


# CART -------------------------------------------------------------------------

# Modelo final -----------------------------------------------------------------
 
CART <- function(df, nombre_modelo, genesKP = 'mezcla'){
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
  
  cart_tun <- tune.rpart(clase ~ . 
                         , data = df_filtrado
                         , minsplit = c(10, 15, 20, 25, 30) 
                         , cp = c(.001, .005, .01, .015, .02, .03, .04)
                         , maxdepth = c(10,15,20,25,30))
  
  cart <- rpart(clase ~ .
                , data = df_filtrado
                , maxsurrogate = 0
                , minsplit = cart_tun$best.parameters$minsplit
                , cp = cart_tun$best.parameters$cp
                , maxdepth = cart_tun$best.parameters$maxdepth )
  
  #Medidas aparentes
  
  predicciones <- predict(cart, newdata = df_filtrado, type = "class")
  
  
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
    , variables = names(cart$variable.importance)
    , modelo = cart
    , tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
  
  
}

# Cálculo del poder predictivo -------------------------------------------------


CART.PP <- function(folds, nombre_modelo, genesKP = "mezcla"){
  tic()
  set.seed(123)
  modRHM <- function(split){
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
    
    
    cart_tun <- tune.rpart(clase ~ . ,
                           data = train_data_filtrado
                           , minsplit = c(10, 15, 20, 25, 30) 
                           , cp = c(.001, .005, .01, .015, .02, .03, .04)
                           , maxdepth = c(10,15,20,25,30))
    
    cart <- rpart(clase ~ .
                  , data = train_data_filtrado
                  , maxsurrogate = 0
                  , minsplit = cart_tun$best.parameters$minsplit
                  , cp = cart_tun$best.parameters$cp
                  , maxdepth = cart_tun$best.parameters$maxdepth )
    
    predicciones <- predict(cart, newdata = test_data_filtrado, type = "class")
    
    
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
  
  resultados_folds <- lapply(folds$splits, modRHM)
  
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



# BRAIN ------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M3.CART.Brain.Grupo1.Final <- CART(df = datos_brain, nombre_modelo = "CART Brain Grupo1 Final", genesKP = "grupo_1")
M3.CART.Brain.Grupo1.PP <- CART.PP(folds = folds_brain, nombre_modelo = "CART Brain Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M3.CART.Brain.Grupo2.Final <- CART(df = datos_brain,nombre_modelo = "CART Brain Grupo2 Final",genesKP = "grupo_2")
M3.CART.Brain.Grupo2.PP    <- CART.PP(folds = folds_brain,nombre_modelo = "CART Brain Grupo2 PP",genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M3.CART.Brain.Grupo3.Final <- CART(df = datos_brain,nombre_modelo = "CART Brain Grupo3 Final",genesKP = "grupo_3")
M3.CART.Brain.Grupo3.PP    <- CART.PP(folds = folds_brain,nombre_modelo = "CART Brain Grupo3 PP",genesKP = "grupo_3")

# SKIN -------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M3.CART.Skin.Grupo1.Final <- CART(df = datos_skin, nombre_modelo = "CART Skin Grupo1 Final", genesKP = "grupo_1")
M3.CART.Skin.Grupo1.PP    <- CART.PP(folds = folds_skin, nombre_modelo = "CART Skin Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------
M3.CART.Skin.Grupo2.Final <- CART(df = datos_skin, nombre_modelo = "CART Skin Grupo2 Final", genesKP = "grupo_2")
M3.CART.Skin.Grupo2.PP    <- CART.PP(folds = folds_skin, nombre_modelo = "CART Skin Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------
M3.CART.Skin.Grupo3.Final <- CART(df = datos_skin, nombre_modelo = "CART Skin Grupo3 Final", genesKP = "grupo_3")
M3.CART.Skin.Grupo3.PP    <- CART.PP(folds = folds_skin, nombre_modelo = "CART Skin Grupo3 PP", genesKP = "grupo_3")


# LUNG -------------------------------------------------------------------------
## Grupo 1 ---------------------------------------------------------------------
M3.CART.Lung.Grupo1.Final <- CART(df = datos_lung, nombre_modelo = "CART Lung Grupo1 Final", genesKP = "grupo_1")
M3.CART.Lung.Grupo1.PP    <- CART.PP(folds = folds_lung, nombre_modelo = "CART Lung Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------
M3.CART.Lung.Grupo2.Final <- CART(df = datos_lung, nombre_modelo = "CART Lung Grupo2 Final", genesKP = "grupo_2")
M3.CART.Lung.Grupo2.PP    <- CART.PP(folds = folds_lung, nombre_modelo = "CART Lung Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------
M3.CART.Lung.Grupo3.Final <- CART(df = datos_lung, nombre_modelo = "CART Lung Grupo3 Final", genesKP = "grupo_3")
M3.CART.Lung.Grupo3.PP    <- CART.PP(folds = folds_lung, nombre_modelo = "CART Lung Grupo3 PP", genesKP = "grupo_3")


# STOMACH ----------------------------------------------------------------------
## Grupo 1 ---------------------------------------------------------------------
M3.CART.Stomach.Grupo1.Final <- CART(df = datos_stomach, nombre_modelo = "CART Stomach Grupo1 Final", genesKP = "grupo_1")
M3.CART.Stomach.Grupo1.PP    <- CART.PP(folds = folds_stomach, nombre_modelo = "CART Stomach Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------
M3.CART.Stomach.Grupo2.Final <- CART(df = datos_stomach, nombre_modelo = "CART Stomach Grupo2 Final", genesKP = "grupo_2")
M3.CART.Stomach.Grupo2.PP    <- CART.PP(folds = folds_stomach, nombre_modelo = "CART Stomach Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------
M3.CART.Stomach.Grupo3.Final <- CART(df = datos_stomach, nombre_modelo = "CART Stomach Grupo3 Final", genesKP = "grupo_3")
M3.CART.Stomach.Grupo3.PP    <- CART.PP(folds = folds_stomach, nombre_modelo = "CART Stomach Grupo3 PP", genesKP = "grupo_3")


# THYROID ----------------------------------------------------------------------
## Grupo 1 ---------------------------------------------------------------------
M3.CART.Thyroid.Grupo1.Final <- CART(df = datos_thyroid, nombre_modelo = "CART Thyroid Grupo1 Final", genesKP = "grupo_1")
M3.CART.Thyroid.Grupo1.PP    <- CART.PP(folds = folds_thyroid, nombre_modelo = "CART Thyroid Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------
M3.CART.Thyroid.Grupo2.Final <- CART(df = datos_thyroid, nombre_modelo = "CART Thyroid Grupo2 Final", genesKP = "grupo_2")
M3.CART.Thyroid.Grupo2.PP    <- CART.PP(folds = folds_thyroid, nombre_modelo = "CART Thyroid Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------
M3.CART.Thyroid.Grupo3.Final <- CART(df = datos_thyroid, nombre_modelo = "CART Thyroid Grupo3 Final", genesKP = "grupo_3")
M3.CART.Thyroid.Grupo3.PP    <- CART.PP(folds = folds_thyroid, nombre_modelo = "CART Thyroid Grupo3 PP", genesKP = "grupo_3")


# BREAST -----------------------------------------------------------------------
## Grupo 1 ---------------------------------------------------------------------
M3.CART.Breast.Grupo1.Final <- CART(df = datos_breast, nombre_modelo = "CART Breast Grupo1 Final", genesKP = "grupo_1")
M3.CART.Breast.Grupo1.PP    <- CART.PP(folds = folds_breast, nombre_modelo = "CART Breast Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------
M3.CART.Breast.Grupo2.Final <- CART(df = datos_breast, nombre_modelo = "CART Breast Grupo2 Final", genesKP = "grupo_2")
M3.CART.Breast.Grupo2.PP    <- CART.PP(folds = folds_breast, nombre_modelo = "CART Breast Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------
M3.CART.Breast.Grupo3.Final <- CART(df = datos_breast, nombre_modelo = "CART Breast Grupo3 Final", genesKP = "grupo_3")
M3.CART.Breast.Grupo3.PP    <- CART.PP(folds = folds_breast, nombre_modelo = "CART Breast Grupo3 PP", genesKP = "grupo_3")


resultados_cart <- list(
  
  # Brain
  M3.CART.Brain.Grupo1.Final = M3.CART.Brain.Grupo1.Final,
  M3.CART.Brain.Grupo1.PP = M3.CART.Brain.Grupo1.PP,
  
  M3.CART.Brain.Grupo2.Final = M3.CART.Brain.Grupo2.Final,
  M3.CART.Brain.Grupo2.PP = M3.CART.Brain.Grupo2.PP,
  
  M3.CART.Brain.Grupo3.Final = M3.CART.Brain.Grupo3.Final,
  M3.CART.Brain.Grupo3.PP = M3.CART.Brain.Grupo3.PP,
  
  
  # Skin
  M3.CART.Skin.Grupo1.Final = M3.CART.Skin.Grupo1.Final,
  M3.CART.Skin.Grupo1.PP = M3.CART.Skin.Grupo1.PP,
  
  M3.CART.Skin.Grupo2.Final = M3.CART.Skin.Grupo2.Final,
  M3.CART.Skin.Grupo2.PP = M3.CART.Skin.Grupo2.PP,
  
  M3.CART.Skin.Grupo3.Final = M3.CART.Skin.Grupo3.Final,
  M3.CART.Skin.Grupo3.PP = M3.CART.Skin.Grupo3.PP,
  
  
  # Lung
  M3.CART.Lung.Grupo1.Final = M3.CART.Lung.Grupo1.Final,
  M3.CART.Lung.Grupo1.PP = M3.CART.Lung.Grupo1.PP,
  
  M3.CART.Lung.Grupo2.Final = M3.CART.Lung.Grupo2.Final,
  M3.CART.Lung.Grupo2.PP = M3.CART.Lung.Grupo2.PP,
  
  M3.CART.Lung.Grupo3.Final = M3.CART.Lung.Grupo3.Final,
  M3.CART.Lung.Grupo3.PP = M3.CART.Lung.Grupo3.PP,
  
  
  # Stomach
  M3.CART.Stomach.Grupo1.Final = M3.CART.Stomach.Grupo1.Final,
  M3.CART.Stomach.Grupo1.PP = M3.CART.Stomach.Grupo1.PP,
  
  M3.CART.Stomach.Grupo2.Final = M3.CART.Stomach.Grupo2.Final,
  M3.CART.Stomach.Grupo2.PP = M3.CART.Stomach.Grupo2.PP,
  
  M3.CART.Stomach.Grupo3.Final = M3.CART.Stomach.Grupo3.Final,
  M3.CART.Stomach.Grupo3.PP = M3.CART.Stomach.Grupo3.PP,
  
  
  # Thyroid
  M3.CART.Thyroid.Grupo1.Final = M3.CART.Thyroid.Grupo1.Final,
  M3.CART.Thyroid.Grupo1.PP = M3.CART.Thyroid.Grupo1.PP,
  
  M3.CART.Thyroid.Grupo2.Final = M3.CART.Thyroid.Grupo2.Final,
  M3.CART.Thyroid.Grupo2.PP = M3.CART.Thyroid.Grupo2.PP,
  
  M3.CART.Thyroid.Grupo3.Final = M3.CART.Thyroid.Grupo3.Final,
  M3.CART.Thyroid.Grupo3.PP = M3.CART.Thyroid.Grupo3.PP,
  
  
  # Breast
  M3.CART.Breast.Grupo1.Final = M3.CART.Breast.Grupo1.Final,
  M3.CART.Breast.Grupo1.PP = M3.CART.Breast.Grupo1.PP,
  
  M3.CART.Breast.Grupo2.Final = M3.CART.Breast.Grupo2.Final,
  M3.CART.Breast.Grupo2.PP = M3.CART.Breast.Grupo2.PP,
  
  M3.CART.Breast.Grupo3.Final = M3.CART.Breast.Grupo3.Final,
  M3.CART.Breast.Grupo3.PP = M3.CART.Breast.Grupo3.PP
)



save(resultados_cart, file = "resultados/resultados_cart.RData")










































































