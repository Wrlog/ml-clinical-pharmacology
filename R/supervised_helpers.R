# -----------------------------------------------------------------------------
# Shared helpers for the supervised SSRI-response analyses (02 and 03).
#
# Each analysis trains four classifiers (random forest, linear SVM, RBF SVM,
# gradient boosting) on two feature sets, tunes them with repeated stratified
# 10-fold CV, evaluates once on the held-out test set, and exports everything
# the dashboard needs as tidy CSVs.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidymodels)
  library(tidyverse)
  library(factoextra)
  library(vip)
  library(future)
})
tidymodels_prefer()
theme_set(theme_bw())

model_labels <- c(
  rf         = "Random forest",
  svm_linear = "SVM (linear)",
  svm_rbf    = "SVM (RBF)",
  gbm        = "Gradient boosting"
)

model_specs <- function() {
  list(
    rf = rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
      set_mode("classification") %>%
      set_engine("ranger"),
    svm_linear = svm_linear(cost = tune()) %>%
      set_mode("classification") %>%
      set_engine("kernlab"),
    svm_rbf = svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
      set_mode("classification") %>%
      set_engine("kernlab"),
    gbm = boost_tree(trees = 1000, tree_depth = tune(), min_n = tune(),
                     loss_reduction = tune(), sample_size = tune(),
                     mtry = tune(), learn_rate = tune()) %>%
      set_mode("classification") %>%
      set_engine("xgboost")
  )
}

# Predictor columns (plus the outcome) a recipe actually uses; needed for
# permutation importance, which works on raw training columns.
recipe_columns <- function(rec) {
  rec$var_info %>% filter(role %in% c("predictor", "outcome")) %>% pull(variable)
}

variable_importance <- function(model, spec, best, rec, final_wf, train) {
  set.seed(123)
  if (model %in% c("rf", "gbm")) {
    # Model-based importance: ranger permutation importance / xgboost gain
    engine_args <- if (model == "rf") list("ranger", importance = "permutation") else list("xgboost")
    vip_spec <- do.call(set_engine, c(list(finalize_model(spec, best)), engine_args))
    fit_obj <- workflow() %>% add_recipe(rec) %>% add_model(vip_spec) %>% fit(train) %>% extract_fit_parsnip()
    set.seed(123)
    vip::vi(fit_obj, scale = TRUE)
  } else {
    # SVMs have no native importance: permute each predictor (10 times) and
    # measure the mean drop in accuracy on the training data
    fit_obj <- final_wf %>% fit(train) %>% extract_fit_parsnip()
    train_cols <- train[, recipe_columns(rec)]
    set.seed(123)
    vip::vi(fit_obj,
       method = "permute",
       target = "response",
       metric = "accuracy",
       pred_wrapper = function(object, newdata) predict(object, newdata)$.pred_class,
       train = as.data.frame(train_cols),
       nsim = 10,
       scale = TRUE,
       type = "difference")
  }
}

fit_evaluate <- function(model, feature_set, rec, split, folds, select_metric) {
  message(sprintf("[%s] %s / %s", format(Sys.time(), "%H:%M:%S"), model_labels[[model]], feature_set))
  spec <- model_specs()[[model]]
  train <- training(split)
  wf <- workflow() %>% add_recipe(rec) %>% add_model(spec)

  set.seed(123)
  tuned <- tune_grid(wf, resamples = folds, grid = 5,
                     metrics = metric_set(roc_auc, accuracy))
  best <- select_best(tuned, metric = select_metric)
  final_wf <- finalize_workflow(wf, best)

  cv <- collect_metrics(tuned) %>%
    filter(.config == best$.config) %>%
    select(.metric, mean, std_err)

  set.seed(123)
  train_auc <- fit(final_wf, train) %>%
    augment(train) %>%
    roc_auc(response, .pred_1, event_level = "second") %>%
    pull(.estimate)

  set.seed(123)
  preds <- last_fit(final_wf, split) %>% collect_predictions()
  cm <- conf_mat(preds, truth = response, estimate = .pred_class)
  # Remission (response = 1, the second factor level) is the positive class
  test <- summary(cm, event_level = "second") %>%
    bind_rows(roc_auc(preds, response, .pred_1, event_level = "second"))

  tag <- function(d) mutate(d, model = model, feature_set = feature_set, .before = 1)
  list(
    metrics = tag(tibble(
      cv_roc_auc = cv$mean[cv$.metric == "roc_auc"],
      cv_roc_auc_se = cv$std_err[cv$.metric == "roc_auc"],
      cv_accuracy = cv$mean[cv$.metric == "accuracy"],
      train_roc_auc = train_auc
    ) %>% bind_cols(test %>% select(.metric, .estimate) %>% pivot_wider(names_from = .metric, values_from = .estimate))),
    params = tag(best %>% select(-.config) %>% mutate(across(everything(), as.character)) %>%
                   pivot_longer(everything(), names_to = "parameter", values_to = "value")),
    roc = tag(roc_curve(preds, response, .pred_1, event_level = "second")),
    confusion = tag(as_tibble(cm$table)),
    # A model that predicts a single class has zero importance everywhere, which
    # scaling turns into NaN; report those as 0
    importance = tag(variable_importance(model, spec, best, rec, final_wf, train) %>%
                       select(Variable, Importance) %>%
                       mutate(Importance = if_else(is.finite(Importance), Importance, 0)))
  )
}

# Accuracy vs. no-information rate (share of the majority class in the test
# set), one-sided exact binomial test as reported by caret::confusionMatrix.
add_nir_test <- function(metrics, test) {
  n_test <- nrow(test)
  nir_test <- max(table(test$response)) / n_test
  metrics %>% mutate(
    nir = nir_test,
    p_value_acc_gt_nir = map_dbl(accuracy, ~ binom.test(round(.x * n_test), n_test, p = nir_test, alternative = "greater")$p.value)
  )
}

run_pca <- function(train, cols, feature_set, out_dir, fig_dir) {
  pc <- prcomp(train[, cols], scale. = TRUE, retx = TRUE)
  eig <- get_eigenvalue(pc)
  p <- fviz_pca_biplot(pc, repel = TRUE, label = "var", habillage = factor(train$response),
                       col.var = "grey20", alpha.ind = 0.6, title = paste("PCA -", feature_set))
  ggsave(file.path(fig_dir, paste0("pca_biplot_", feature_set, ".png")), p, width = 7, height = 5, dpi = 150, bg = "white")
  list(
    loadings = as_tibble(pc$rotation, rownames = "variable") %>% mutate(feature_set = feature_set, .before = 1),
    variance = tibble(feature_set = feature_set, component = rownames(eig), eig),
    scores = tibble(feature_set = feature_set, PC1 = pc$x[, 1], PC2 = pc$x[, 2], response = train$response)
  )
}

plot_importance <- function(importance, fig_dir) {
  importance %>%
    group_by(model, feature_set) %>%
    group_walk(function(d, key) {
      p <- ggplot(d, aes(x = Importance, y = fct_reorder(Variable, Importance))) +
        geom_col(fill = "#3b6ea8") +
        labs(y = NULL, x = "Scaled importance",
             title = paste0(model_labels[[key$model]], " - ", key$feature_set))
      ggsave(file.path(fig_dir, paste0("importance_", key$model, "_", key$feature_set, ".png")),
             p, width = 5, height = 4, dpi = 150, bg = "white")
    })
}

plot_roc <- function(roc, fig_dir) {
  p <- roc %>%
    mutate(model = model_labels[model]) %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity, colour = model)) +
    geom_abline(linetype = "dashed", colour = "grey60") +
    geom_path(linewidth = 0.8) +
    coord_equal() +
    facet_wrap(~feature_set) +
    labs(colour = NULL, title = "Test-set ROC curves") +
    theme(legend.position = "bottom")
  ggsave(file.path(fig_dir, "roc_curves.png"), p, width = 9, height = 5, dpi = 150, bg = "white")
}

run_supervised_analysis <- function(split, recipes, pca_sets, select_metric, out_dir) {
  fig_dir <- file.path(out_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
  train <- training(split)
  test <- testing(split)

  # Exploratory PCA on the training data
  pcas <- imap(pca_sets, ~ run_pca(train, .x, .y, out_dir, fig_dir))
  write_csv(map_dfr(pcas, "loadings"), file.path(out_dir, "pca_loadings.csv"))
  write_csv(map_dfr(pcas, "variance"), file.path(out_dir, "pca_variance.csv"))
  write_csv(map_dfr(pcas, "scores"), file.path(out_dir, "pca_scores.csv"))

  set.seed(123)
  folds <- vfold_cv(train, v = 10, repeats = 5, strata = response)

  runs <- expand_grid(model = names(model_labels), feature_set = names(recipes)) %>%
    pmap(function(model, feature_set) {
      fit_evaluate(model, feature_set, recipes[[feature_set]], split, folds, select_metric)
    })
  collect <- function(part) map_dfr(runs, part)

  metrics <- collect("metrics") %>% add_nir_test(test)
  write_csv(metrics, file.path(out_dir, "metrics.csv"))
  write_csv(collect("params"), file.path(out_dir, "best_params.csv"))
  write_csv(collect("roc"), file.path(out_dir, "roc_curves.csv"))
  write_csv(collect("confusion"), file.path(out_dir, "confusion_matrices.csv"))
  write_csv(collect("importance"), file.path(out_dir, "variable_importance.csv"))

  plot_importance(collect("importance"), fig_dir)
  plot_roc(collect("roc"), fig_dir)

  print(metrics %>% select(model, feature_set, cv_roc_auc, roc_auc, accuracy, sens, spec, p_value_acc_gt_nir), width = Inf)
  invisible(metrics)
}
