# -----------------------------------------------------------------------------
# 01 - Unsupervised learning on gene expression
#
# Question: do unsupervised clusters of 10-gene expression profiles recover the
# clinical diagnosis (healthy control / mild / moderate / treatment-resistant
# depression)?  i.e. do the clusters have "diagnostic validity"?
#
# Methods: density inspection, PCA, k-means (k = 3-5), hierarchical clustering
# (complete / single / average linkage) and Gaussian mixture models (mclust).
#
# Input : data/gene_expression.csv  (ID, 10 genes, Diagnosis; n = 30)
# Output: results/01_unsupervised/
# Run from the repository root:  Rscript analysis/01_unsupervised_gene_expression.R
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggdendro)
  library(mclust)
  library(factoextra)
  library(reshape2)
})

out_dir <- "results/01_unsupervised"
fig_dir <- file.path(out_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

save_png <- function(p, name, w, h) {
  ggsave(file.path(fig_dir, paste0(name, ".png")), p, width = w, height = h, dpi = 150, bg = "white")
}

df <- read.csv("data/gene_expression.csv", stringsAsFactors = FALSE, header = TRUE)
genes <- setdiff(names(df), c("ID", "Diagnosis"))
dat <- as.matrix(df[, genes])

### Probability density inspection ###
pdfData <- melt(df[, c(genes, "Diagnosis")], id.vars = "Diagnosis")

p <- ggplot(pdfData, aes(x = value, group = factor(Diagnosis), color = factor(Diagnosis))) +
  geom_density(alpha = 0.7, adjust = 2) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(x = "Gene expression", y = "Probability density", color = "Diagnosis") +
  facet_wrap(~variable, ncol = 5)
save_png(p, "density_by_gene", 9, 5)

p <- ggplot(pdfData, aes(x = value, group = factor(variable), color = factor(Diagnosis))) +
  geom_density(alpha = 0.7, adjust = 2) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(x = "Gene expression", y = "Probability density", color = "Diagnosis") +
  facet_wrap(~Diagnosis, ncol = 4)
save_png(p, "density_by_diagnosis", 9, 3.5)

### Differentially expressed genes ###
p <- ggplot(pdfData, aes(x = factor(Diagnosis), y = value, fill = factor(Diagnosis))) +
  geom_boxplot() +
  theme_bw() +
  theme(legend.position = "bottom", axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  labs(x = NULL, y = "Gene expression", fill = "Diagnosis") +
  facet_wrap(~variable, ncol = 5)
save_png(p, "expression_boxplots", 9, 5.5)

### Principal component analysis ###
pc_df <- prcomp(dat, scale. = TRUE, retx = TRUE)
eig <- get_eigenvalue(pc_df)
write.csv(data.frame(component = rownames(eig), eig, row.names = NULL),
          file.path(out_dir, "pca_variance.csv"), row.names = FALSE)
write.csv(data.frame(gene = rownames(pc_df$rotation), pc_df$rotation, row.names = NULL),
          file.path(out_dir, "pca_loadings.csv"), row.names = FALSE)
save_png(fviz_pca_biplot(pc_df, repel = TRUE, habillage = factor(df$Diagnosis)), "pca_biplot", 7, 5)

### Clustering ###
# Each run stores its assignments so the dashboard can colour the PCA plot
# by any clustering, and so cluster/diagnosis agreement can be scored.
assignments <- data.frame(ID = df$ID, Diagnosis = df$Diagnosis,
                          PC1 = pc_df$x[, 1], PC2 = pc_df$x[, 2])

pc_cluster_plot <- function(cluster, title) {
  ggplot(assignments, aes(x = PC1, y = PC2, colour = factor(cluster), shape = Diagnosis)) +
    geom_point(size = 3) +
    labs(x = "Component 1", y = "Component 2", colour = "Cluster", title = title) +
    theme_bw()
}

# k-means: elbow curve for choosing k
wss <- (nrow(dat) - 1) * sum(apply(dat, 2, var))
set.seed(1000)
for (i in 2:20) wss[i] <- sum(kmeans(dat, centers = i, nstart = 25)$withinss)
write.csv(data.frame(k = 1:20, within_ss = wss), file.path(out_dir, "kmeans_elbow.csv"), row.names = FALSE)

for (k in 3:5) {
  set.seed(1000)
  cl <- kmeans(dat, centers = k)$cluster
  assignments[[paste0("kmeans_", k)]] <- cl
  save_png(pc_cluster_plot(cl, paste0("k-means, k = ", k)), paste0("pc_kmeans_", k), 6, 4)
}

# Hierarchical clustering on the Euclidean distance matrix
d <- dist(dat, method = "euclidean")
for (linkage in c("complete", "single", "average")) {
  hc <- hclust(d, method = linkage)
  hc$labels <- paste0(df$ID, " ", substr(df$Diagnosis, 1, 1))
  assignments[[paste0("hclust_", linkage, "_4")]] <- cutree(hc, k = 4)
  p <- ggdendrogram(hc) + labs(title = paste0(tools::toTitleCase(linkage), " linkage"))
  save_png(p, paste0("hc_", linkage, "_linkage"), 5, 6)
}

# Gaussian mixture models: BIC-selected number of components, then fixed G
gmm_auto <- Mclust(dat)
assignments$gmm_auto <- gmm_auto$classification
save_png(pc_cluster_plot(gmm_auto$classification, paste0("GMM, BIC-selected (G = ", gmm_auto$G, ")")),
         "pc_gmm_auto", 6, 4)
for (g in 3:5) {
  cl <- Mclust(dat, G = g)$classification
  assignments[[paste0("gmm_", g)]] <- cl
  save_png(pc_cluster_plot(cl, paste0("GMM, G = ", g)), paste0("pc_gmm_", g), 6, 4)
}

gmm_5 <- Mclust(dat, G = 5)
clusterProbability <- data.frame(ID = df$ID, round(gmm_5$z, digits = 3))
colnames(clusterProbability) <- c("ID", paste("Cluster", 1:5))
write.csv(clusterProbability, file.path(out_dir, "gmm_cluster_probability.csv"), row.names = FALSE)

write.csv(assignments, file.path(out_dir, "cluster_assignments.csv"), row.names = FALSE)

### Diagnostic validity: agreement between clusters and diagnosis ###
cluster_cols <- setdiff(names(assignments), c("ID", "Diagnosis", "PC1", "PC2"))
agreement <- data.frame(
  clustering = cluster_cols,
  n_clusters = sapply(cluster_cols, function(x) length(unique(assignments[[x]]))),
  adjusted_rand_index = sapply(cluster_cols, function(x) adjustedRandIndex(assignments[[x]], df$Diagnosis)),
  row.names = NULL
)
write.csv(agreement, file.path(out_dir, "cluster_diagnosis_agreement.csv"), row.names = FALSE)
print(agreement)

cat("GMM BIC-selected model:", gmm_auto$modelName, "with G =", gmm_auto$G, "\n")
