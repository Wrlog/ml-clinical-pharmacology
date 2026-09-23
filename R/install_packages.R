# Install the R packages used by the analyses (R >= 4.3).
pkgs <- c("tidymodels", "tidyverse", "future", "parallelly", "ranger", "kernlab", "xgboost",
          "vip", "factoextra", "FactoMineR", "mclust", "ggdendro", "reshape2", "jsonlite")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
# vip is occasionally unavailable as a CRAN binary; fall back to the source archive
if (!requireNamespace("vip", quietly = TRUE)) {
  install.packages("https://cran.r-project.org/src/contrib/Archive/vip/vip_0.4.6.tar.gz", repos = NULL, type = "source")
}
