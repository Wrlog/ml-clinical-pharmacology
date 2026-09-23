# Reproduce every result and rebuild the dashboard data. Run from the repository root:
#   Rscript run_all.R
for (script in c("analysis/01_unsupervised_gene_expression.R",
                 "analysis/02_ssri_response_prediction.R",
                 "analysis/03_ssri_response_with_symptoms.R",
                 "R/build_dashboard_data.R")) {
  message("== ", script)
  local(source(script, local = TRUE))
}
