# -----------------------------------------------------------------------------
# 03 - Predicting SSRI remission with baseline symptom-level data
#
# Question: extends analysis 02 with item-level baseline depressive symptoms
# (QIDS-SR items). Do clinical + symptom features predict remission, and do
# biological measures add predictive value on top of them?
#
# Design: predefined split (300 training patients from one site, 100 external
# test patients); 4 classifiers x 2 feature sets tuned with 5x repeated
# stratified 10-fold CV (hyperparameters chosen by ROC AUC).
#
# Input : data/ssri_cohort_qids.csv
# Output: results/03_ssri_response_symptoms/
# Run from the repository root:  Rscript analysis/03_ssri_response_with_symptoms.R
# -----------------------------------------------------------------------------

source("R/supervised_helpers.R")
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

biological <- c("plasmaSerotonin", "AHR", "TSPAN5", "CYP2C19", "DEFB1", "SLC6A4", "COMT", "CYP2D6", "ERICH3")

df <- read.csv("data/ssri_cohort_qids.csv", header = TRUE, stringsAsFactors = FALSE) %>%
  select(-X) %>%
  mutate(response = as.factor(response))

symptoms <- grep("^QIDS", names(df), value = TRUE)
clinical <- c("age", "maritalStatus", "smokingStatus", "numChildren", "seasonalDepression",
              "socioEconomicStatus", "MDD_severity", symptoms)

# Use the predefined train/test assignment rather than a random split
split <- make_splits(
  list(analysis = which(df$train_test == "Train"), assessment = which(df$train_test == "Test")),
  data = df %>% select(-train_test)
)
train <- training(split)

recipes <- list(
  clinical = recipe(response ~ ., data = train) %>%
    update_role(ID, new_role = "subject ID") %>%
    update_role(all_of(biological), new_role = "biological variable"),
  clinical_biomarkers = recipe(response ~ ., data = train) %>%
    update_role(ID, new_role = "subject ID")
)

run_supervised_analysis(
  split = split,
  recipes = recipes,
  pca_sets = list(clinical = clinical, clinical_biomarkers = c(clinical, biological)),
  select_metric = "roc_auc",
  out_dir = "results/03_ssri_response_symptoms"
)
