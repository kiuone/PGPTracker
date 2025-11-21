#!/usr/bin/env Rscript

# NSTI Calculation using Castor
# Nearest Sequenced Taxon Index for study sequences

library(castor, quietly = TRUE)

Args <- commandArgs(TRUE)

# Command-line arguments
tree_file <- Args[1]
known_tips_file <- Args[2]
output_file <- Args[3]

# Read tree
full_tree <- read_tree(file=tree_file, check_label_uniqueness = TRUE)

# Read known tips (reference sequences)
known_tips <- read.table(known_tips_file, header=FALSE, stringsAsFactors = FALSE)$V1

# Identify unknown tips (study sequences)
unknown_tips_index <- which(! full_tree$tip.label %in% known_tips)
unknown_tips <- full_tree$tip.label[unknown_tips_index]
known_tip_range <- which(! full_tree$tip.label %in% unknown_tips)

# Calculate NSTI (nearest phylogenetic distance)
nsti_calc <- find_nearest_tips(
    full_tree,
    target_tips=known_tip_range,
    check_input=TRUE
)

nsti_values <- nsti_calc$nearest_distance_per_tip[unknown_tips_index]
nsti_genomes <- full_tree$tip.label[nsti_calc$nearest_tip_per_tip[unknown_tips_index]]

# Format output
nsti_df <- data.frame(
    sequence = unknown_tips,
    metadata_NSTI = nsti_values,
    closest_reference_genome = nsti_genomes
)

# Write results
write.table(nsti_df, file=output_file, sep="\t", quote=FALSE, row.names=FALSE)
