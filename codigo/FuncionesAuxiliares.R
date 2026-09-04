#-------------------------  Funciónes Auxiliares -------------------------------


# Paqueterias ------------------------------------------------------------------

library(dplyr)
library(metrica)
library(caret)
library(rpart.plot)
library(ppcor)
library(ggcorrplot)
library(ggplot2)
library(GGally)
library(cowplot)
library(RColorBrewer)
library(ggbiplot)


# Funciónes Auxiliares ---------------------------------------------------------


# Matriz de dispersión ---------------------------------------------------------
matriz_dispersion <- function(df, cancer, genesKP = "grupo_1",
                              titulo = "Diagramas de dispersión",
                              subtitulo = "Principales enzimas involucradas en la vía de las Kinureninas",
                              mostrar_grafica = TRUE){
  
  # Filtramos variables de interés
  
  if(genesKP == "grupo_1"){
    df_filtrado <- df[ , c(grupo_1, "clase")]
    genes_usados <- grupo_1
  } else if(genesKP == "grupo_2"){
    df_filtrado <- df[ , c(grupo_2, "clase")]
    genes_usados <- grupo_2
  } else if(genesKP == "grupo_3"){
    df_filtrado <- df[ , c(grupo_3, "clase")]
    genes_usados <- grupo_3
  } else {
    stop("genesKP debe ser 'grupo_1', 'grupo_2' o 'grupo_3'")
  }
  
  
  
  df_filtrado$clase <- droplevels(as.factor(df_filtrado$clase))
  
  
  estrellas_p <- function(p){
    if(is.na(p)) return("")
    if(p < 0.001) return("***")
    if(p < 0.01)  return("**")
    if(p < 0.05)  return("*")
    return("")
  }
  
  
  panel_cor_clase <- function(data, mapping, ...){
    
    x <- GGally::eval_data_col(data, mapping$x)
    y <- GGally::eval_data_col(data, mapping$y)
    
    
    clase <- droplevels(data$clase)
    clases <- levels(clase)
    
    
    
    colores <- RColorBrewer::brewer.pal(
      max(3, length(clases)),
      "Set2"
    )[1:length(clases)]
    
    
    oscurecer_color <- function(color, factor = 0.65){
      
      rgb_color <- grDevices::col2rgb(color)
      
      grDevices::rgb(
        red = rgb_color[1, ] * factor,
        green = rgb_color[2, ] * factor,
        blue = rgb_color[3, ] * factor,
        maxColorValue = 255
      )
    }
    
    colores_oscuros <- oscurecer_color(colores)
    
    
    formatear_r <- function(r){
      if(is.na(r)) return("NA")
      formatC(r, format = "f", digits = 2)
    }
    
    
    # Correlación global
    
    cor_global <- suppressWarnings(
      cor(
        x,
        y,
        use = "complete.obs",
        method = "pearson"
      )
    )
    
    p_global <- tryCatch(
      cor.test(
        x,
        y,
        method = "pearson"
      )$p.value,
      error = function(e) NA_real_
    )
    
    etiqueta_global <- paste0(
      "r = ",
      formatear_r(cor_global),
      estrellas_p(p_global)
    )
    
    
    # Correlaciones por clase
    
    
    etiquetas_abreviadas <- c("G", "T")
    
    etiquetas_clase <- character(length(clases))
    
    for(i in seq_along(clases)){
      
      cl <- clases[i]
      idx <- clase == cl
      
      cor_clase <- suppressWarnings(
        cor(
          x[idx],
          y[idx],
          use = "complete.obs",
          method = "pearson"
        )
      )
      
      p_clase <- tryCatch(
        cor.test(
          x[idx],
          y[idx],
          method = "pearson"
        )$p.value,
        error = function(e) NA_real_
      )
      
      etiquetas_clase[i] <- paste0(
        etiquetas_abreviadas[i],
        ": ",
        formatear_r(cor_clase),
        estrellas_p(p_clase)
      )
    }
    
    

    p <- ggplot() +
      xlim(0, 1) +
      ylim(0, 1) +
      theme_void()
    
    
    # Correlación global
    p <- p +
      annotate(
        "text",
        x = 0.03,
        y = 0.78,
        label = etiqueta_global,
        hjust = 0,
        size = 3,
        colour = "black"
      )
    
    
    # Primera clase
    if(length(clases) >= 1){
      
      p <- p +
        annotate(
          "text",
          x = 0.03,
          y = 0.50,
          label = etiquetas_clase[1],
          hjust = 0,
          size = 3,
          colour = colores_oscuros[1]
        )
    }
    
    
    # Segunda clase
    if(length(clases) >= 2){
      
      p <- p +
        annotate(
          "text",
          x = 0.03,
          y = 0.22,
          label = etiquetas_clase[2],
          hjust = 0,
          size = 3,
          colour = colores_oscuros[2]
        )
    }
    
    
    return(p)
  }
  
  
  # Matriz de dispersión
  
  Aux1 <- GGally::ggpairs(
    data = df_filtrado,
    columns = 1:length(genes_usados),
    mapping = aes(
      color = clase,
      fill = clase
    ),
    lower = list(
      continuous = GGally::wrap(
        "points",
        alpha = 0.5,
        size = 0.5
      )
    ),
    diag = list(
      continuous = GGally::wrap(
        "densityDiag",
        alpha = 0.6
      )
    ),
    upper = list(
      continuous = panel_cor_clase
    )
  ) +
    scale_color_brewer(
      palette = "Set2",
      name = "Clase"
    ) +
    scale_fill_brewer(
      palette = "Set2",
      name = "Clase"
    ) +
    labs(
      title = titulo,
      subtitle = subtitulo
    ) +
    theme_bw() +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 18),
      plot.subtitle = element_text(size = 14),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.x = element_blank(),
      axis.ticks.y = element_blank()
    )
  
  
  # Gráfica auxiliar para extraer la leyenda
  
  Aux2 <- ggplot(
    df_filtrado,
    aes(
      x = 1,
      y = 1,
      color = clase
    )
  ) +
    geom_point() +
    scale_color_brewer(
      palette = "Set2",
      name = "Clase"
    ) +
    theme_minimal() +
    theme(
      legend.position = "right",
      axis.title = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )
  
  
  # Extraemos leyenda y convertimos matriz
  
  legenda <- cowplot::get_legend(Aux2)
  
  Diagrama <- GGally::ggmatrix_gtable(Aux1)
  
  
  # Combinamos matriz y leyenda
  
  grafica_final <- cowplot::plot_grid(
    Diagrama,
    legenda,
    rel_widths = c(6, 1)
  )
  
  
  # Mostramos la gráfica
  
  if(mostrar_grafica){
    print(grafica_final)
  }
  
  
  return(
    list(
      grafica = grafica_final,
      matriz_base = Aux1,
      datos = df_filtrado,
      cancer = cancer,
      grupo = genesKP
    )
  )
}


## Kruskal Wallis --------------------------------------------------------------

analisis_kruskal <- function(df, cancer, genesKP = "grupo_1",
                             titulo = "Kruskal-Wallis Test",
                             subtitulo = NULL,
                             ylab = "Expresión génica",
                             mostrar_grafica = TRUE,
                             ncol = 4){
  
  # Filtramos variables de interés
  
  if(genesKP == "grupo_1"){
    df_filtrado <- df[ , c(grupo_1, "clase")]
    genes_usados <- grupo_1
  } else if(genesKP == "grupo_2"){
    df_filtrado <- df[ , c(grupo_2, "clase")]
    genes_usados <- grupo_2
  } else if(genesKP == "grupo_3"){
    df_filtrado <- df[ , c(grupo_3, "clase")]
    genes_usados <- grupo_3
  } else {
    stop("genesKP debe ser 'grupo_1', 'grupo_2' o 'grupo_3'")
  }
  
  
  datos_largos <- df_filtrado %>%
    tidyr::pivot_longer(
      cols = -clase,
      names_to = "gen",
      values_to = "value"
    ) %>%
    dplyr::mutate(
      clase = as.factor(clase),
      gen = factor(gen, levels = genes_usados)
    )
  
  
  # Kruskal-Wallis por gen
  
  resultados_kw <- datos_largos %>%
    dplyr::group_by(gen) %>%
    dplyr::summarise(
      p_value = kruskal.test(value ~ clase)$p.value,
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      etiqueta = paste0(
        "K-W p = ",
        formatC(p_value, format = "e", digits = 2)
      )
    )
  
  
  # Tabla descriptiva
  
  tabla_descriptivos <- datos_largos %>%
    dplyr::group_by(gen, clase) %>%
    dplyr::summarise(
      mediana = median(value, na.rm = TRUE),
      min = min(value, na.rm = TRUE),
      max = max(value, na.rm = TRUE),
      p25 = quantile(value, 0.25, na.rm = TRUE),
      p75 = quantile(value, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::left_join(
      resultados_kw %>% dplyr::select(gen, p_value),
      by = "gen"
    ) %>%
    dplyr::mutate(
      cancer = cancer
    ) %>%
    dplyr::select(
      cancer, gen, clase, mediana, min, max, p25, p75, p_value
    )
  
  
  
  if(is.null(subtitulo)){
    subtitulo <- paste(cancer, "|", genesKP)
  }
  
  
  
  
  grafica <- ggplot(datos_largos, aes(x = clase, y = value)) +
    
    geom_jitter(
      aes(color = clase),
      width = 0.2,
      alpha = 0.5,
      size = 1.2
    ) +
    
    geom_boxplot(
      aes(fill = clase),
      alpha = 0.6,
      outlier.shape = NA,
      width = 0.6
    ) +
    
    geom_text(
      data = resultados_kw,
      aes(
        x = .5,
        y = Inf,
        label = etiqueta
      ),
      inherit.aes = FALSE,
      hjust = 0,
      vjust = 1.4,
      colour = "steelblue4",
      size = 3.5
    ) +
    
    facet_wrap(~ gen, scales = "free_y", ncol = ncol) +
    
    scale_fill_brewer(palette = "Set2") +
    scale_color_brewer(palette = "Set2") +
    
    labs(
      x = "Clase",
      y = ylab,
      title = titulo,
      subtitle = subtitulo
    ) +
    
    theme_bw() +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(face = "plain"),
      strip.text = element_text(face = "bold"),
      axis.text.x = element_text(angle = 0, hjust = 0.5)
    )
  

  
  if(mostrar_grafica){
    print(grafica)
  }
  
  
  return(list(
    grafica = grafica,
    tabla = tabla_descriptivos,
    kruskal = resultados_kw
  ))
}


# Heat Map Matriz de correlaciones parciales -----------------------------------

correlacion_parcial <- function(df, cancer, genesKP = "grupo_1",
                                titulo = "Matriz de correlaciones parciales",
                                mostrar_graficas = TRUE,
                                digits = 2,
                                lab_size = 5){
  
  
  # Filtramos variables de interés
  
  if(genesKP == "grupo_1"){
    df_filtrado <- df[ , c(grupo_1, "clase")]
    genes_usados <- grupo_1
  } else if(genesKP == "grupo_2"){
    df_filtrado <- df[ , c(grupo_2, "clase")]
    genes_usados <- grupo_2
  } else if(genesKP == "grupo_3"){
    df_filtrado <- df[ , c(grupo_3, "clase")]
    genes_usados <- grupo_3
  } else {
    stop("genesKP debe ser 'grupo_1', 'grupo_2' o 'grupo_3'")
  }
  
  
  
  df_filtrado$clase <- as.factor(df_filtrado$clase)
  clases <- levels(df_filtrado$clase)
  
  
  matrices <- list()
  graficas <- list()
  
  
  for(cl in clases){
    
    datos_clase <- df_filtrado %>%
      dplyr::filter(clase == cl) %>%
      dplyr::select(-clase)
    
    
    datos_clase <- datos_clase[ , genes_usados]
    
    
    # Correlación parcial de Pearson
    corr_parcial <- ppcor::pcor(datos_clase, method = "pearson")
    
    matriz_corr <- corr_parcial$estimate %>%
      round(digits = digits) %>%
      as.matrix()
    
    
    # Guardamos matriz
    matrices[[cl]] <- matriz_corr
    
    
    # Creamos gráfica
    grafica_actual <- ggcorrplot::ggcorrplot(
      matriz_corr,
      colors = c("#6D9EC1", "white", "#E46726"),
      type = "lower",
      lab = TRUE,
      lab_size = lab_size,
      lab_col = "black"
    ) +
      labs(
        title = titulo,
        subtitle = paste("Grupo", cl),
        x = "",
        y = ""
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(size = 18),
        plot.subtitle = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(
          size = 12,
          angle = 45,
          hjust = 1
        ),
        axis.text.y = element_text(size = 12)
      )
    
    
    # Guardamos gráfica
    graficas[[cl]] <- grafica_actual
    
    
    # La mostramos en Plots
    if(mostrar_graficas){
      print(grafica_actual)
    }
  }
  
  
  return(list(
    graficas = graficas,
    matrices = matrices,
    cancer = cancer,
    grupo = genesKP,
    clases = clases
  ))
}

# Componentes Principales ------------------------------------------------------

componentes_principales <- function(df, cancer, genesKP = "grupo_1",
                                    titulo = "Componentes Principales",
                                    subtitulo = "Análisis de la varianza explicada por los componentes",
                                    mostrar_graficas = TRUE,
                                    limites = NULL){
  
  # Filtramos variables de interés
  
  if(genesKP == "grupo_1"){
    df_filtrado <- df[ , c(grupo_1, "clase")]
    genes_usados <- grupo_1
  } else if(genesKP == "grupo_2"){
    df_filtrado <- df[ , c(grupo_2, "clase")]
    genes_usados <- grupo_2
  } else if(genesKP == "grupo_3"){
    df_filtrado <- df[ , c(grupo_3, "clase")]
    genes_usados <- grupo_3
  } else {
    stop("genesKP debe ser 'grupo_1', 'grupo_2' o 'grupo_3'")
  }
  
  
  
  df_filtrado$clase <- droplevels(as.factor(df_filtrado$clase))
  

  
  datos_numericos <- df_filtrado %>%
    dplyr::select(-clase)
  
  
  
  pca <- prcomp(
    datos_numericos,
    center = TRUE,
    scale. = FALSE
  )
  
  
  var_exp <- summary(pca)$importance[2, 1:3] * 100
  
  
  
  tabla_varianza <- data.frame(
    componente = c("PC1", "PC2", "PC3"),
    varianza_explicada = as.numeric(var_exp)
  )
  
  
  crear_biplot <- function(comp_x, comp_y){
    
    grafica <- ggbiplot::ggbiplot(
      pca,
      group = df_filtrado$clase,
      choices = c(comp_x, comp_y),
      ellipse = TRUE,
      ellipse.alpha = 0.0
    ) +
      scale_color_brewer(
        palette = "Set2",
        name = "Clase"
      ) +
      scale_fill_brewer(
        palette = "Set2",
        name = "Clase"
      ) +
      guides(
        fill = "none",
        color = guide_legend(title = "Clase")
      ) +
      labs(
        title = titulo,
        subtitle = subtitulo,
        x = paste0(
          "PC", comp_x,
          " (", round(var_exp[comp_x], 2), "%)"
        ),
        y = paste0(
          "PC", comp_y,
          " (", round(var_exp[comp_y], 2), "%)"
        )
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(size = 18),
        plot.subtitle = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)
      )
    
    
    
    if(!is.null(limites)){
      
      grafica <- grafica +
        coord_cartesian(
          xlim = limites,
          ylim = limites
        )
    }
    
    
    return(grafica)
  }
  
  
  
  grafica_PC1_PC2 <- crear_biplot(
    comp_x = 1,
    comp_y = 2
  )
  
  grafica_PC1_PC3 <- crear_biplot(
    comp_x = 1,
    comp_y = 3
  )
  
  grafica_PC2_PC3 <- crear_biplot(
    comp_x = 2,
    comp_y = 3
  )
  
  
  # Mostramos las gráficas
  
  if(mostrar_graficas){
    
    print(grafica_PC1_PC2)
    print(grafica_PC1_PC3)
    print(grafica_PC2_PC3)
  }
  
  
  
  return(
    list(
      PC1_PC2 = grafica_PC1_PC2,
      PC1_PC3 = grafica_PC1_PC3,
      PC2_PC3 = grafica_PC2_PC3,
      pca = pca,
      varianza = tabla_varianza,
      datos = df_filtrado,
      cancer = cancer,
      grupo = genesKP
    )
  )
}

# Grafica CART -----------------------------------------------------------------

graficar_CART <- function(mod, titulo){
  X11()
  prp(mod$modelo,
      type = 4, 
      clip.right.labs = FALSE, 
      extra = 101, 
      under = TRUE, 
      under.cex = 1, 
      fallen.leaves = TRUE, 
      box.palette = "GnYlRd", 
      cex=.7)
  
  title(titulo, line = 3, cex.main = 1)
}



# Evaluacion aparente ---------------------------------------------------------- 
evaluacion_aparente <- function(mod, datos, X_matrix = NULL){
  
  newdf <- if (is.null(X_matrix)) datos else X_matrix
  
  if (is.null(X_matrix)){
    predicciones <- predict(mod, newdata=newdf, type = "response")
  } else {
    predicciones <- predict(mod, newx=newdf, type = "response")
  }
  
  predicciones_clase <- ifelse( predicciones >= .5
                                , levels(datos$clase)[2] #Positivo
                                , levels(datos$clase)[1] #Negativo 
                                )
  
  
  matriz <- confusionMatrix(data = factor(predicciones_clase, levels = levels(datos$clase))
                            , reference = datos$clase
                            , positive = levels(datos$clase)[2]
  )
  
  respond <- data.frame(
    Metric = c("accuracy", "recall", "specificity")
    , Score = c(
      as.numeric(matriz$overall["Accuracy"])
      , as.numeric(matriz$byClass["Sensitivity"])
      , as.numeric(matriz$byClass["Specificity"])
    )
  )
  
  
  
  return(list(
    metricas = respond
    , matrizConfusion = matriz$table
    
  ))
  
  
}

# Evaluacion correcta ----------------------------------------------------------

evaluacion_correcta <- function(mod, test_df, train_df){
  predicciones <- predict(mod, newdata=test_df, type = "response")
  predicciones_clase <- ifelse( predicciones >= .5
                                , levels(train_df$clase)[2] #Positivo
                                , levels(train_df$clase)[1] #Negativo 
  )
  
  
  matriz <- confusionMatrix(
    data = factor(predicciones_clase, levels = levels(train_df$clase))
    , reference = test_df$clase
    , positive = levels(train_df$clase)[2]
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


# Evaluacion correcta 2 --------------------------------------------------------

evaluacion_correcta2 <- function(mod, train_df, X_test, Y_test){
  predicciones <- predict(mod, newx=X_test, type = "response")
  predicciones_clase <- ifelse( predicciones >= .5
                                , levels(train_df$clase)[2] #Positivo
                                , levels(train_df$clase)[1] #Negativo 
  )
  
  
  matriz <- confusionMatrix(
    data = factor(predicciones_clase, levels = levels(train_df$clase))
    , reference = Y_test
    , positive = levels(train_df$clase)[2]
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
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  