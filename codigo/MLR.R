

# Regresión Logistica  Multinomial ---------------------------------------------

# Paqueterias ------------------------------------------------------------------

library(plyr)
library(dplyr)
library(tidyr)
library(forcats)
library(rsample)
library(tictoc)
library(VGAM)
library(glmnet)

library(ggplot2)

library(readr)

# Funciones Auxiliares ---------------------------------------------------------

source("codigo/FuncionesAuxiliares.R")


# Conjuntos Train y Test -------------------------------------------------------

load("datos/Folds.RData")


# Regresión logistica simple ---------------------------------------------------

# Modelo final 

MLR1 <- function(df, nombre_modelo, genesKP = "mezcla"){
  tic()
  
  # Filtramos variables de interes
  
  if(genesKP == "grupo_1"){
    df_filtrado <- df[ , c(grupo_1, 'clase')]
  } else if (genesKP == "grupo_2"){
    df_filtrado <- df[ , c(grupo_2, 'clase')]
  } else if (genesKP == "grupo_3"){
    df_filtrado <- df[ , c(grupo_3, 'clase')]
  } else if (genesKP == "mezcla") {
    df_filtrado <- df
  }
  
  modelo <- glm(
    clase ~ . 
    , data = df_filtrado
    , family = binomial("logit")
  )
  
  evaluacion <- evaluacion_aparente(mod = modelo,  datos = df_filtrado, X_matrix = NULL)
  
  tiempo <- toc(quiet = TRUE)
  
  return(list(
    nombreModelo = nombre_modelo
    , metricasAparentes = evaluacion$metricas
    , matrizConfusion = evaluacion$matrizConfusion
    , coeficientes = coef(modelo)
    , modelo = modelo
    , tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
  
}

# Cálculo del poder predictivo 

MLR1.PP <- function(folds, nombre_modelo, genesKP = "mezcla"){
  tic()
  modKCV <- function(split){
    # Train y test
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
    
    
    modtr <- glm(clase ~ ., data = train_data_filtrado, family = binomial(link = "logit"))
    
    
    evaluacion <- evaluacion_correcta(mod = modtr, test_df = test_data_filtrado, train_df = train_data_filtrado)
    
    return(evaluacion)
  }
  
  resultados_folds <- lapply(folds$splits, modKCV)
  
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



# Regresión logistica con regularización Lasso -----------------------------------


# Modelo final 

MLR2 <- function(df, nombre_modelo, genesKP = "mezcla"){
  tic()
  # Filtramos variables de interes
  
  if(genesKP == "grupo_1"){
    df_filtrado <- df[ , c(grupo_1, 'clase')]
  } else if (genesKP == "grupo_2"){
    df_filtrado <- df[ , c(grupo_2, 'clase')]
  } else if (genesKP == "grupo_3"){
    df_filtrado <- df[ , c(grupo_3, 'clase')]
  } else if (genesKP == "mezcla") {
    df_filtrado <- df
  }
  
  X_mod <- model.matrix(clase ~ .^2, data=df_filtrado)[,-1]
  Y_mod <- df_filtrado$clase
  
  set.seed(123)
  
  cv_mod <- cv.glmnet(X_mod
                      ,Y_mod
                      , family = "binomial"
                      , alpha = 1
                      , type.measure = "class")
  
  mod_final <- glmnet(X_mod
                      , Y_mod
                      , family = "binomial"
                      , alpha = 1
                      , lambda = cv_mod$lambda.min)
  
  coeficientes <- as.data.frame(as.matrix(coef(mod_final)[-1,1]))
  
  colnames(coeficientes) <- "Coeficientes"
  
  coeficientes <- subset(coeficientes, Coeficientes != 0)
  
  evaluacion <- evaluacion_aparente(mod=mod_final, datos=df_filtrado, X_matrix = X_mod)
  
  tiempo <- toc(quiet = TRUE)
  
  return(list(
    nombreModelo = nombre_modelo
    , metricasAparentes = evaluacion$metricas
    , matrizConfusion = evaluacion$matrizConfusion
    , coeficientes = coeficientes
    , modelo = mod_final
    , tiempoEjecucion = tiempo$toc - tiempo$tic
  ))
  
  
}


# Cálculo del poder predictivo 

MLR2.PP <- function(folds, nombre_modelo, genesKP = "mezcla"){
  tic()
  mod2KCV <- function(split){
    # Train y test
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
    

    x_train <- model.matrix(clase ~ .^2, data = train_data_filtrado)[, -1]
    y_train <- train_data_filtrado$clase
    x_test <- model.matrix(clase ~ .^2, data = test_data_filtrado)[, -1]
    y_test <- test_data_filtrado$clase
    
    
    cv_fit <- cv.glmnet(x_train, y_train, 
                        family = "binomial", 
                        alpha = 1,
                        type.measure = "class")
    

    mod_final <- glmnet(x_train, y_train, 
                        family = "binomial", 
                        alpha = 1,
                        lambda = cv_fit$lambda.min)
    

    evaluacion <- evaluacion_correcta2(mod = mod_final, train_df = train_data_filtrado, X_test = x_test, Y_test = y_test)
    
    return(evaluacion)
  }
  
  resultados_folds <- lapply(folds$splits, mod2KCV)
  
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

M1.MLR.Brain.Grupo1.Final <-  MLR1(df = datos_brain, nombre_modelo = "MLR Brain Grupo1 Final", genesKP = "grupo_1")
M2.MLR.Brain.Grupo1.Final <-  MLR2(df = datos_brain, nombre_modelo = "MLR Lasso Brain Grupo1 Final", genesKP = "grupo_1")

M1.MLR.Brain.Grupo1.PP <-  MLR1.PP(folds = folds_brain , nombre_modelo = "MLR Brain Grupo1 PP", genesKP = "grupo_1")
M2.MLR.Brain.Grupo1.PP <-  MLR2.PP(folds = folds_brain , nombre_modelo = "MLR Lasso Brain Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M1.MLR.Brain.Grupo2.Final <-  MLR1(df = datos_brain, nombre_modelo = "MLR Brain Grupo2 Final", genesKP = "grupo_2")
M2.MLR.Brain.Grupo2.Final <-  MLR2(df = datos_brain, nombre_modelo = "MLR Lasso Brain Grupo2 Final", genesKP = "grupo_2")

M1.MLR.Brain.Grupo2.PP <-  MLR1.PP(folds = folds_brain , nombre_modelo = "MLR Brain Grupo2 PP", genesKP = "grupo_2")
M2.MLR.Brain.Grupo2.PP <-  MLR2.PP(folds = folds_brain , nombre_modelo = "MLR Lasso Brain Grupo2 PP", genesKP = "grupo_2")

## Grupo 3 ---------------------------------------------------------------------

M1.MLR.Brain.Grupo3.Final <-  MLR1(df = datos_brain, nombre_modelo = "MLR Brain Grupo3 Final", genesKP = "grupo_3")
M2.MLR.Brain.Grupo3.Final <-  MLR2(df = datos_brain, nombre_modelo = "MLR Lasso Brain Grupo3 Final", genesKP = "grupo_3")

M1.MLR.Brain.Grupo3.PP <-  MLR1.PP(folds = folds_brain , nombre_modelo = "MLR Brain Grupo3 PP", genesKP = "grupo_3")
M2.MLR.Brain.Grupo3.PP <-  MLR2.PP(folds = folds_brain , nombre_modelo = "MLR Lasso Brain Grupo3 PP", genesKP = "grupo_3")


# SKIN -------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M1.MLR.Skin.Grupo1.Final <-  MLR1(df = datos_skin, nombre_modelo = "MLR Skin Grupo1 Final", genesKP = "grupo_1")
M2.MLR.Skin.Grupo1.Final <-  MLR2(df = datos_skin, nombre_modelo = "MLR Lasso Skin Grupo1 Final", genesKP = "grupo_1")

M1.MLR.Skin.Grupo1.PP <-  MLR1.PP(folds = folds_skin , nombre_modelo = "MLR Skin Grupo1 PP", genesKP = "grupo_1")
M2.MLR.Skin.Grupo1.PP <-  MLR2.PP(folds = folds_skin , nombre_modelo = "MLR Lasso Skin Grupo1 PP", genesKP = "grupo_1")

## Grupo 2 ---------------------------------------------------------------------

M1.MLR.Skin.Grupo2.Final <-  MLR1(df = datos_skin, nombre_modelo = "MLR Skin Grupo2 Final", genesKP = "grupo_2")
M2.MLR.Skin.Grupo2.Final <-  MLR2(df = datos_skin, nombre_modelo = "MLR Lasso Skin Grupo2 Final", genesKP = "grupo_2")

M1.MLR.Skin.Grupo2.PP <-  MLR1.PP(folds = folds_skin , nombre_modelo = "MLR Skin Grupo2 PP", genesKP = "grupo_2")
M2.MLR.Skin.Grupo2.PP <-  MLR2.PP(folds = folds_skin , nombre_modelo = "MLR Lasso Skin Grupo2 PP", genesKP = "grupo_2")


## Grupo 3 ---------------------------------------------------------------------

M1.MLR.Skin.Grupo3.Final <-  MLR1(df = datos_skin, nombre_modelo = "MLR Skin Grupo3 Final", genesKP = "grupo_3")
M2.MLR.Skin.Grupo3.Final <-  MLR2(df = datos_skin, nombre_modelo = "MLR Lasso Skin Grupo3 Final", genesKP = "grupo_3")

M1.MLR.Skin.Grupo3.PP <-  MLR1.PP(folds = folds_skin , nombre_modelo = "MLR Skin Grupo3 PP", genesKP = "grupo_3")
M2.MLR.Skin.Grupo3.PP <-  MLR2.PP(folds = folds_skin , nombre_modelo = "MLR Lasso Skin Grupo3 PP", genesKP = "grupo_3")


# STOMACH ----------------------------------------------------------------------

## Grupo 1 -------------------------------------------------------------

M1.MLR.Stomach.Grupo1.Final <-  MLR1(df = datos_stomach, nombre_modelo = "MLR Stomach Grupo1 Final", genesKP = "grupo_1")
M2.MLR.Stomach.Grupo1.Final <-  MLR2(df = datos_stomach, nombre_modelo = "MLR Lasso Stomach Grupo1 Final", genesKP = "grupo_1")

M1.MLR.Stomach.Grupo1.PP <-  MLR1.PP(folds = folds_stomach , nombre_modelo = "MLR Stomach Grupo1 PP", genesKP = "grupo_1")
M2.MLR.Stomach.Grupo1.PP <-  MLR2.PP(folds = folds_stomach , nombre_modelo = "MLR Lasso Stomach Grupo1 PP", genesKP = "grupo_1")


## Grupo 2 -------------------------------------------------------------

M1.MLR.Stomach.Grupo2.Final <-  MLR1(df = datos_stomach, nombre_modelo = "MLR Stomach Grupo2 Final", genesKP = "grupo_2")
M2.MLR.Stomach.Grupo2.Final <-  MLR2(df = datos_stomach, nombre_modelo = "MLR Lasso Stomach Grupo2 Final", genesKP = "grupo_2")

M1.MLR.Stomach.Grupo2.PP <-  MLR1.PP(folds = folds_stomach , nombre_modelo = "MLR Stomach Grupo2 PP", genesKP = "grupo_2")
M2.MLR.Stomach.Grupo2.PP <-  MLR2.PP(folds = folds_stomach , nombre_modelo = "MLR Lasso Stomach Grupo2 PP", genesKP = "grupo_2")


## Grupo 3 -------------------------------------------------------------

M1.MLR.Stomach.Grupo3.Final <-  MLR1(df = datos_stomach, nombre_modelo = "MLR Stomach Grupo3 Final", genesKP = "grupo_3")
M2.MLR.Stomach.Grupo3.Final <-  MLR2(df = datos_stomach, nombre_modelo = "MLR Lasso Stomach Grupo3 Final", genesKP = "grupo_3")

M1.MLR.Stomach.Grupo3.PP <-  MLR1.PP(folds = folds_stomach , nombre_modelo = "MLR Stomach Grupo3 PP", genesKP = "grupo_3")
M2.MLR.Stomach.Grupo3.PP <-  MLR2.PP(folds = folds_stomach , nombre_modelo = "MLR Lasso Stomach Grupo3 PP", genesKP = "grupo_3")


# LUNG -------------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M1.MLR.Lung.Grupo1.Final <-  MLR1(df = datos_lung, nombre_modelo = "MLR Lung Grupo1 Final", genesKP = "grupo_1")
M2.MLR.Lung.Grupo1.Final <-  MLR2(df = datos_lung, nombre_modelo = "MLR Lasso Lung Grupo1 Final", genesKP = "grupo_1")

M1.MLR.Lung.Grupo1.PP <-  MLR1.PP(folds = folds_lung , nombre_modelo = "MLR Lung Grupo1 PP", genesKP = "grupo_1")
M2.MLR.Lung.Grupo1.PP <-  MLR2.PP(folds = folds_lung , nombre_modelo = "MLR Lasso Lung Grupo1 PP", genesKP = "grupo_1")


## Grupo 2 ---------------------------------------------------------------------

M1.MLR.Lung.Grupo2.Final <-  MLR1(df = datos_lung, nombre_modelo = "MLR Lung Grupo2 Final", genesKP = "grupo_2")
M2.MLR.Lung.Grupo2.Final <-  MLR2(df = datos_lung, nombre_modelo = "MLR Lasso Lung Grupo2 Final", genesKP = "grupo_2")

M1.MLR.Lung.Grupo2.PP <-  MLR1.PP(folds = folds_lung , nombre_modelo = "MLR Lung Grupo2 PP", genesKP = "grupo_2")
M2.MLR.Lung.Grupo2.PP <-  MLR2.PP(folds = folds_lung , nombre_modelo = "MLR Lasso Lung Grupo2 PP", genesKP = "grupo_2")


## Grupo 3 ---------------------------------------------------------------------

M1.MLR.Lung.Grupo3.Final <-  MLR1(df = datos_lung, nombre_modelo = "MLR Lung Grupo3 Final", genesKP = "grupo_3")
M2.MLR.Lung.Grupo3.Final <-  MLR2(df = datos_lung, nombre_modelo = "MLR Lasso Lung Grupo3 Final", genesKP = "grupo_3")

M1.MLR.Lung.Grupo3.PP <-  MLR1.PP(folds = folds_lung , nombre_modelo = "MLR Lung Grupo3 PP", genesKP = "grupo_3")
M2.MLR.Lung.Grupo3.PP <-  MLR2.PP(folds = folds_lung , nombre_modelo = "MLR Lasso Lung Grupo3 PP", genesKP = "grupo_3")

# BREAST -----------------------------------------------------------------------

## Grupo 1 ---------------------------------------------------------------------

M1.MLR.Breast.Grupo1.Final <-  MLR1(df = datos_breast, nombre_modelo = "MLR Breast Grupo1 Final", genesKP = "grupo_1")
M2.MLR.Breast.Grupo1.Final <-  MLR2(df = datos_breast, nombre_modelo = "MLR Lasso Breast Grupo1 Final", genesKP = "grupo_1")

M1.MLR.Breast.Grupo1.PP <-  MLR1.PP(folds = folds_breast , nombre_modelo = "MLR Breast Grupo1 PP", genesKP = "grupo_1")
M2.MLR.Breast.Grupo1.PP <-  MLR2.PP(folds = folds_breast , nombre_modelo = "MLR Lasso Breast Grupo1 PP", genesKP = "grupo_1")


## Grupo 2 ---------------------------------------------------------------------

M1.MLR.Breast.Grupo2.Final <-  MLR1(df = datos_breast, nombre_modelo = "MLR Breast Grupo2 Final", genesKP = "grupo_2")
M2.MLR.Breast.Grupo2.Final <-  MLR2(df = datos_breast, nombre_modelo = "MLR Lasso Breast Grupo2 Final", genesKP = "grupo_2")

M1.MLR.Breast.Grupo2.PP <-  MLR1.PP(folds = folds_breast , nombre_modelo = "MLR Breast Grupo2 PP", genesKP = "grupo_2")
M2.MLR.Breast.Grupo2.PP <-  MLR2.PP(folds = folds_breast , nombre_modelo = "MLR Lasso Breast Grupo2 PP", genesKP = "grupo_2")


## Grupo 3 ---------------------------------------------------------------------

M1.MLR.Breast.Grupo3.Final <-  MLR1(df = datos_breast, nombre_modelo = "MLR Breast Grupo3 Final", genesKP = "grupo_3")
M2.MLR.Breast.Grupo3.Final <-  MLR2(df = datos_breast, nombre_modelo = "MLR Lasso Breast Grupo3 Final", genesKP = "grupo_3")

M1.MLR.Breast.Grupo3.PP <-  MLR1.PP(folds = folds_breast , nombre_modelo = "MLR Breast Grupo3 PP", genesKP = "grupo_3")
M2.MLR.Breast.Grupo3.PP <-  MLR2.PP(folds = folds_breast , nombre_modelo = "MLR Lasso Breast Grupo3 PP", genesKP = "grupo_3")

# THYROID ----------------------------------------------------------------------

## Grupo 1 -------------------------------------------------------------

M1.MLR.Thyroid.Grupo1.Final <-  MLR1(df = datos_thyroid, nombre_modelo = "MLR Thyroid Grupo1 Final", genesKP = "grupo_1")
M2.MLR.Thyroid.Grupo1.Final <-  MLR2(df = datos_thyroid, nombre_modelo = "MLR Lasso Thyroid Grupo1 Final", genesKP = "grupo_1")

M1.MLR.Thyroid.Grupo1.PP <-  MLR1.PP(folds = folds_thyroid , nombre_modelo = "MLR Thyroid Grupo1 PP", genesKP = "grupo_1")
M2.MLR.Thyroid.Grupo1.PP <-  MLR2.PP(folds = folds_thyroid , nombre_modelo = "MLR Lasso Thyroid Grupo1 PP", genesKP = "grupo_1")


## Grupo 2 -------------------------------------------------------------

M1.MLR.Thyroid.Grupo2.Final <-  MLR1(df = datos_thyroid, nombre_modelo = "MLR Thyroid Grupo2 Final", genesKP = "grupo_2")
M2.MLR.Thyroid.Grupo2.Final <-  MLR2(df = datos_thyroid, nombre_modelo = "MLR Lasso Thyroid Grupo2 Final", genesKP = "grupo_2")

M1.MLR.Thyroid.Grupo2.PP <-  MLR1.PP(folds = folds_thyroid , nombre_modelo = "MLR Thyroid Grupo2 PP", genesKP = "grupo_2")
M2.MLR.Thyroid.Grupo2.PP <-  MLR2.PP(folds = folds_thyroid , nombre_modelo = "MLR Lasso Thyroid Grupo2 PP", genesKP = "grupo_2")


## Grupo 3 -------------------------------------------------------------

M1.MLR.Thyroid.Grupo3.Final <-  MLR1(df = datos_thyroid, nombre_modelo = "MLR Thyroid Grupo3 Final", genesKP = "grupo_3")
M2.MLR.Thyroid.Grupo3.Final <-  MLR2(df = datos_thyroid, nombre_modelo = "MLR Lasso Thyroid Grupo3 Final", genesKP = "grupo_3")

M1.MLR.Thyroid.Grupo3.PP <-  MLR1.PP(folds = folds_thyroid , nombre_modelo = "MLR Thyroid Grupo3 PP", genesKP = "grupo_3")
M2.MLR.Thyroid.Grupo3.PP <-  MLR2.PP(folds = folds_thyroid , nombre_modelo = "MLR Lasso Thyroid Grupo3 PP", genesKP = "grupo_3")





resultados_mlr <- list(
  
  # Brain
  M1.MLR.Brain.Grupo1.Final = M1.MLR.Brain.Grupo1.Final,
  M2.MLR.Brain.Grupo1.Final = M2.MLR.Brain.Grupo1.Final,
  M1.MLR.Brain.Grupo1.PP = M1.MLR.Brain.Grupo1.PP,
  M2.MLR.Brain.Grupo1.PP = M2.MLR.Brain.Grupo1.PP,
  
  M1.MLR.Brain.Grupo2.Final = M1.MLR.Brain.Grupo2.Final,
  M2.MLR.Brain.Grupo2.Final = M2.MLR.Brain.Grupo2.Final,
  M1.MLR.Brain.Grupo2.PP = M1.MLR.Brain.Grupo2.PP,
  M2.MLR.Brain.Grupo2.PP = M2.MLR.Brain.Grupo2.PP,
  
  M1.MLR.Brain.Grupo3.Final = M1.MLR.Brain.Grupo3.Final,
  M2.MLR.Brain.Grupo3.Final = M2.MLR.Brain.Grupo3.Final,
  M1.MLR.Brain.Grupo3.PP = M1.MLR.Brain.Grupo3.PP,
  M2.MLR.Brain.Grupo3.PP = M2.MLR.Brain.Grupo3.PP,
  
  
  # Skin
  M1.MLR.Skin.Grupo1.Final = M1.MLR.Skin.Grupo1.Final,
  M2.MLR.Skin.Grupo1.Final = M2.MLR.Skin.Grupo1.Final,
  M1.MLR.Skin.Grupo1.PP = M1.MLR.Skin.Grupo1.PP,
  M2.MLR.Skin.Grupo1.PP = M2.MLR.Skin.Grupo1.PP,
  
  M1.MLR.Skin.Grupo2.Final = M1.MLR.Skin.Grupo2.Final,
  M2.MLR.Skin.Grupo2.Final = M2.MLR.Skin.Grupo2.Final,
  M1.MLR.Skin.Grupo2.PP = M1.MLR.Skin.Grupo2.PP,
  M2.MLR.Skin.Grupo2.PP = M2.MLR.Skin.Grupo2.PP,
  
  M1.MLR.Skin.Grupo3.Final = M1.MLR.Skin.Grupo3.Final,
  M2.MLR.Skin.Grupo3.Final = M2.MLR.Skin.Grupo3.Final,
  M1.MLR.Skin.Grupo3.PP = M1.MLR.Skin.Grupo3.PP,
  M2.MLR.Skin.Grupo3.PP = M2.MLR.Skin.Grupo3.PP,
  
  
  # Lung
  M1.MLR.Lung.Grupo1.Final = M1.MLR.Lung.Grupo1.Final,
  M2.MLR.Lung.Grupo1.Final = M2.MLR.Lung.Grupo1.Final,
  M1.MLR.Lung.Grupo1.PP = M1.MLR.Lung.Grupo1.PP,
  M2.MLR.Lung.Grupo1.PP = M2.MLR.Lung.Grupo1.PP,
  
  M1.MLR.Lung.Grupo2.Final = M1.MLR.Lung.Grupo2.Final,
  M2.MLR.Lung.Grupo2.Final = M2.MLR.Lung.Grupo2.Final,
  M1.MLR.Lung.Grupo2.PP = M1.MLR.Lung.Grupo2.PP,
  M2.MLR.Lung.Grupo2.PP = M2.MLR.Lung.Grupo2.PP,
  
  M1.MLR.Lung.Grupo3.Final = M1.MLR.Lung.Grupo3.Final,
  M2.MLR.Lung.Grupo3.Final = M2.MLR.Lung.Grupo3.Final,
  M1.MLR.Lung.Grupo3.PP = M1.MLR.Lung.Grupo3.PP,
  M2.MLR.Lung.Grupo3.PP = M2.MLR.Lung.Grupo3.PP,
  
  
  # Stomach
  M1.MLR.Stomach.Grupo1.Final = M1.MLR.Stomach.Grupo1.Final,
  M2.MLR.Stomach.Grupo1.Final = M2.MLR.Stomach.Grupo1.Final,
  M1.MLR.Stomach.Grupo1.PP = M1.MLR.Stomach.Grupo1.PP,
  M2.MLR.Stomach.Grupo1.PP = M2.MLR.Stomach.Grupo1.PP,
  
  M1.MLR.Stomach.Grupo2.Final = M1.MLR.Stomach.Grupo2.Final,
  M2.MLR.Stomach.Grupo2.Final = M2.MLR.Stomach.Grupo2.Final,
  M1.MLR.Stomach.Grupo2.PP = M1.MLR.Stomach.Grupo2.PP,
  M2.MLR.Stomach.Grupo2.PP = M2.MLR.Stomach.Grupo2.PP,
  
  M1.MLR.Stomach.Grupo3.Final = M1.MLR.Stomach.Grupo3.Final,
  M2.MLR.Stomach.Grupo3.Final = M2.MLR.Stomach.Grupo3.Final,
  M1.MLR.Stomach.Grupo3.PP = M1.MLR.Stomach.Grupo3.PP,
  M2.MLR.Stomach.Grupo3.PP = M2.MLR.Stomach.Grupo3.PP,
  
  
  # Thyroid
  M1.MLR.Thyroid.Grupo1.Final = M1.MLR.Thyroid.Grupo1.Final,
  M2.MLR.Thyroid.Grupo1.Final = M2.MLR.Thyroid.Grupo1.Final,
  M1.MLR.Thyroid.Grupo1.PP = M1.MLR.Thyroid.Grupo1.PP,
  M2.MLR.Thyroid.Grupo1.PP = M2.MLR.Thyroid.Grupo1.PP,
  
  M1.MLR.Thyroid.Grupo2.Final = M1.MLR.Thyroid.Grupo2.Final,
  M2.MLR.Thyroid.Grupo2.Final = M2.MLR.Thyroid.Grupo2.Final,
  M1.MLR.Thyroid.Grupo2.PP = M1.MLR.Thyroid.Grupo2.PP,
  M2.MLR.Thyroid.Grupo2.PP = M2.MLR.Thyroid.Grupo2.PP,
  
  M1.MLR.Thyroid.Grupo3.Final = M1.MLR.Thyroid.Grupo3.Final,
  M2.MLR.Thyroid.Grupo3.Final = M2.MLR.Thyroid.Grupo3.Final,
  M1.MLR.Thyroid.Grupo3.PP = M1.MLR.Thyroid.Grupo3.PP,
  M2.MLR.Thyroid.Grupo3.PP = M2.MLR.Thyroid.Grupo3.PP,
  
  
  # Breast
  M1.MLR.Breast.Grupo1.Final = M1.MLR.Breast.Grupo1.Final,
  M2.MLR.Breast.Grupo1.Final = M2.MLR.Breast.Grupo1.Final,
  M1.MLR.Breast.Grupo1.PP = M1.MLR.Breast.Grupo1.PP,
  M2.MLR.Breast.Grupo1.PP = M2.MLR.Breast.Grupo1.PP,
  
  M1.MLR.Breast.Grupo2.Final = M1.MLR.Breast.Grupo2.Final,
  M2.MLR.Breast.Grupo2.Final = M2.MLR.Breast.Grupo2.Final,
  M1.MLR.Breast.Grupo2.PP = M1.MLR.Breast.Grupo2.PP,
  M2.MLR.Breast.Grupo2.PP = M2.MLR.Breast.Grupo2.PP,
  
  M1.MLR.Breast.Grupo3.Final = M1.MLR.Breast.Grupo3.Final,
  M2.MLR.Breast.Grupo3.Final = M2.MLR.Breast.Grupo3.Final,
  M1.MLR.Breast.Grupo3.PP = M1.MLR.Breast.Grupo3.PP,
  M2.MLR.Breast.Grupo3.PP = M2.MLR.Breast.Grupo3.PP
)



save(resultados_mlr, file = "resultados/resultados_MLR.RData")







