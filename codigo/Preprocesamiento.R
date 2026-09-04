# -------------------------  Pre-procesamiento   -------------------------------

# Paqueterias ------------------------------------------------------------------

#Se requieren las siguientes paqueterias
#install.packages("BiocManager")
#BiocManager::install("biomaRt")
#BiocManager::install("org.Hs.eg.db")

library(readr)
library(dplyr)
library(biomaRt)
library(corrplot)
library(forcats)
library(rsample)

# Datos ------------------------------------------------------------------------

datos <- read_tsv("datos/denseDataOnlyDownload.tsv")


# Limpieza ---------------------------------------------------------------------

colnames(datos) <- gsub("\\..*", "", colnames(datos))


genes_ensg <- colnames(datos)
genes_ensg <- genes_ensg[grepl("^ENSG", genes_ensg)]
                         
                         
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

mapping <- getBM(
  attributes = c("ensembl_gene_id", "hgnc_symbol"),
  filters = "ensembl_gene_id",
  values = genes_ensg,
  mart = mart
)


map_dict <- setNames(mapping$hgnc_symbol, mapping$ensembl_gene_id)                         
                         
colnames(datos) <- ifelse(colnames(datos) %in% names(map_dict),
                          map_dict[colnames(datos)],
                          colnames(datos))

datos <- datos[, colnames(datos) != ""] # ENSG00000274276.4 PseudoGen de CBS, no lo mapea

colnames(datos)[colnames(datos) == "USP2-AS1"] <- "USP2_AS1"
# Pre-procesamiento ------------------------------------------------------------


datos <- datos %>% 
  
  dplyr::select(-c("sample", "samples")) %>%
  
  mutate(
    TCGA_GTEX_main_category = factor(TCGA_GTEX_main_category),
    detailed_category = factor(detailed_category)
  ) %>% 
  # Filtrar solo los 10 tipos seleccionados
  filter(TCGA_GTEX_main_category %in% c(
    "TCGA Skin Cutaneous Melanoma", "GTEX Skin",
    "TCGA Stomach Adenocarcinoma", "GTEX Stomach",
    "TCGA Thyroid Carcinoma", "GTEX Thyroid",
    "TCGA Breast Invasive Carcinoma", "GTEX Breast",
    "TCGA Lung Adenocarcinoma", "GTEX Lung",
    "GTEX Brain", 
    "TCGA Brain Lower Grade Glioma",
    "TCGA Glioblastoma Multiforme"
  )) %>%
  
  # Filtrar categorías detalladas relacionadas
  filter(detailed_category %in% c(
    "Skin - Sun Exposed (Lower Leg)", "Skin - Not Sun Exposed (Suprapubic)", "Skin Cutaneous Melanoma",
    "Stomach", "Stomach Adenocarcinoma",
    "Thyroid", "Thyroid Carcinoma",
    "Breast - Mammary Tissue", "Breast Invasive Carcinoma",
    "Lung", "Lung Adenocarcinoma",
    "Brain - Cortex",
    "Brain - Anterior Cingulate Cortex (Ba24)",
    "Brain - Frontal Cortex (Ba9)",
    "Brain Lower Grade Glioma",
    "Glioblastoma Multiforme"
  )) %>%
  # Reordenar niveles del factor
  mutate(
    TCGA_GTEX_main_category = fct_relevel(
      droplevels(TCGA_GTEX_main_category),
      c("GTEX Skin", "TCGA Skin Cutaneous Melanoma",
        "GTEX Stomach", "TCGA Stomach Adenocarcinoma",
        "GTEX Thyroid", "TCGA Thyroid Carcinoma",
        "GTEX Breast", "TCGA Breast Invasive Carcinoma",
        "GTEX Lung", "TCGA Lung Adenocarcinoma",
        "GTEX Brain", 
        "TCGA Brain Lower Grade Glioma",
        "TCGA Glioblastoma Multiforme")
    )
  ) %>%
  # Quitar variables que no vamos a usar
  dplyr::select(-c("_gender", "detailed_category")) %>% 
  
  dplyr::rename(y = TCGA_GTEX_main_category) %>% 
  # Crear variable de clases codificadas
  mutate(
    clase = factor(case_when(
      y == "GTEX Skin" ~ "GTEX_SKIN",
      y == "TCGA Skin Cutaneous Melanoma" ~ "SKCM",
      
      y == "GTEX Stomach" ~ "GTEX_STOMACH",
      y == "TCGA Stomach Adenocarcinoma" ~ "STAD",
      
      y == "GTEX Thyroid" ~ "GTEX_THYROID",
      y == "TCGA Thyroid Carcinoma" ~ "THCA",
      
      y == "GTEX Breast" ~ "GTEX_BREAST",
      y == "TCGA Breast Invasive Carcinoma" ~ "BRCA",
      
      y == "GTEX Lung" ~ "GTEX_LUNG",
      y == "TCGA Lung Adenocarcinoma" ~ "LUAD",
      
      y == "GTEX Brain" ~ "GTEX_B",
      y == "TCGA Brain Lower Grade Glioma" ~ "TCGA_B",
      y == "TCGA Glioblastoma Multiforme" ~ "TCGA_B"
    ))
  ) %>%
  
  dplyr::select(-y)


# Selección de variables -------------------------------------------------------
                         
grupo_1 <- c(
  
  "TDO2", "IDO1", "IDO2", "AFMID", "GOT2", "AADAT", 
  "KYNU", "KMO", "QPRT", "HAAO", "ACMSD",
  "SLC7A5",    # Transportador de Tryptophan (LAT1)
  "SLC3A2",    # CD98 - Co-transportador
  "SLC36A4",   # Transportador de Kynurenine
  "KAT2A"      # Kynurenine aminotransferase II
)

grupo_2 <- c(
  "SLC1A1","SLC1A4","SLC1A5","SLC7A5","SLC7A6","SLC7A8","SLC7A11"  
  ,"MAT1A","MAT2A","MAT2B"    
  ,"AHCY","AHCYL1","MTR","BHMT","CTH","CAT","MPST","CDO1","CBS"
)

grupo_3 <- c(
  "USP1","USP2","USP2_AS1","USP5","USP7","USP8","USP10","USP11","USP15","USP22","USP28","USP9X" 
)



# Creación de dataframe por tipo de cáncer -------------------------------------

datos_skin <- datos %>%
  filter(clase %in% c("GTEX_SKIN","SKCM")) %>%
  mutate(clase = factor(clase, levels = c("GTEX_SKIN","SKCM")))


datos_stomach <- datos %>%
  filter(clase %in% c("GTEX_STOMACH","STAD")) %>%
  mutate(clase = factor(clase, levels = c("GTEX_STOMACH","STAD")))

datos_lung <- datos %>%
  filter(clase %in% c("GTEX_LUNG","LUAD")) %>%
  mutate(clase = factor(clase, levels = c("GTEX_LUNG","LUAD")))

datos_breast <- datos %>%
  filter(clase %in% c("GTEX_BREAST","BRCA")) %>%
  mutate(clase = factor(clase, levels = c("GTEX_BREAST","BRCA")))


datos_thyroid <- datos %>%
  filter(clase %in% c("GTEX_THYROID","THCA")) %>%
  mutate(clase = factor(clase, levels = c("GTEX_THYROID","THCA")))

datos_brain <- datos %>%
  filter(clase %in% c("GTEX_B", "TCGA_B")) %>%
  mutate(clase = factor(clase, levels =c("GTEX_B", "TCGA_B")))

# Lista con los df
lista_datos <- list(
  skin = datos_skin,
  stomach = datos_stomach,
  lung = datos_lung,
  breast = datos_breast,
  thyroid = datos_thyroid,
  brain = datos_brain
)

                         
# Remuestreo -------------------------------------------------------------------

set.seed(10011997)
                         
v <- 5
repeats <- 20 

folds_skin <- vfold_cv(data = datos_skin,v = v,repeats = repeats,strata = clase)
folds_stomach <- vfold_cv(data = datos_stomach,v = v,repeats = repeats,strata = clase)
folds_lung <- vfold_cv(data = datos_lung,v = v,repeats = repeats,strata = clase)
folds_breast <- vfold_cv(data = datos_breast,v = v,repeats = repeats,strata = clase)
folds_thyroid <- vfold_cv(data = datos_thyroid,v = v,repeats = repeats,strata = clase)
folds_brain <- vfold_cv(data = datos_brain,v = v,repeats = repeats,strata = clase)                         
                         
                         
# Eliminamos lo no necesario ---------------------------------------------------

rm(datos, genes_ensg, mapping, mart, map_dict)


# Se guarda el entorno ---------------------------------------------------------

save.image("datos/Folds.RData")

                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         