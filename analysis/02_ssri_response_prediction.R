# -----------------------------------------------------------------------------
# 02 - Predicting SSRI remission from sociodemographic and biological data
#
# Question: in 400 patients with major depressive disorder treated with an SSRI,
#   1) do sociodemographic/clinical factors predict remission better than chance?
#   2) does adding biological measures (plasma serotonin + 8 pharmacogenes)
#      improve prediction?
#
# Design: random 75/25 train/test split; 4 classifiers x 2 feature sets tuned
# with 5x repeated stratified 10-fold CV (hyperparameters chosen by accuracy).
#
# Input : data/ssri_cohort.csv
# Output: results/02_ssri_response/
# Run from the repository root:  Rscript analysis/02_ssri_response_prediction.R
# -----------------------------------------------------------------------------

source("R/supervised_helpers.R")
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

biological <- c("plasmaSerotonin", "AHR", "TSPAN5", "CYP2C19", "DEFB1", "SLC6A4", "COMT", "CYP2D6", "ERICH3")
clinical <- c("age", "maritalStatus", "smokingStatus", "numChildren", "seasonalDepression",
              "socioEconomicStatus", "MDD_severity")

df <- read_csv("data/ssri_cohort.csv", show_col_types = FALSE) %>%
  mutate(response = as.factor(response))

set.seed(123)
split <- initial_split(df)
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
  select_metric = "accuracy",
  out_dir = "results/02_ssri_response"
)
