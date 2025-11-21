#!/usr/bin/env Rscript

# Hidden State Prediction using Castor
# Stateless R script for PGPTracker

library(castor, quietly = TRUE)

Args <- commandArgs(TRUE)

# Command-line arguments
tree_file <- Args[1]
trait_file <- Args[2]
output_file <- Args[3]
hsp_method <- Args[4]  # 'mp' (max parsimony) or 'emp_prob' (empirical probability)

# Read tree and trait table
full_tree <- read_tree(file=tree_file, check_label_uniqueness = TRUE)
full_tree$edge.length[which(full_tree$edge.length == 0)] <- 0.00001
trait_values <- read.delim(trait_file, check.names=FALSE, row.names=1)

# Fix quoted tip labels if present
if(length(grep("\"", full_tree$tip.label) > 0) || length(grep("\'", full_tree$tip.label) > 0)) {
    unknown_tips_index <- which(! full_tree$tip.label %in% rownames(trait_values))
    unknown_tips <- full_tree$tip.label[unknown_tips_index]
    unknown_labels_no_quotes <- gsub("\'", "", unknown_tips)
    unknown_labels_no_quotes <- gsub("\"", "", unknown_labels_no_quotes)
    no_quote_matches = which(unknown_labels_no_quotes %in% rownames(trait_values))

    if(length(no_quote_matches) > 0) {
        indices_to_change <- unknown_tips_index[no_quote_matches]
        full_tree$tip.label[indices_to_change] <- unknown_labels_no_quotes[no_quote_matches]
    }
}

# Identify unknown tips (study sequences not in reference)
unknown_tips_index <- which(! full_tree$tip.label %in% rownames(trait_values))
unknown_tips <- full_tree$tip.label[unknown_tips_index]
num_unknown <- length(unknown_tips)

# Create NA entries for unknown tips
unknown_df <- as.data.frame(matrix(NA, nrow=num_unknown, ncol=ncol(trait_values)))
rownames(unknown_df) = unknown_tips
colnames(unknown_df) = colnames(trait_values)

# Combine known and unknown tips
trait_values <- rbind(trait_values, unknown_df)
remove(unknown_df)
invisible(gc(verbose = FALSE))

# Order trait table to match tree tip labels
trait_values <- trait_values[full_tree$tip.label, , drop=FALSE]

# Run HSP based on method
if (hsp_method == "mp") {
    # Add 1 to counts (states must start at 1)
    trait_values <- trait_values + 1

    hsp_results <- lapply(trait_values, function(trait) {
        mp_hsp <- hsp_max_parsimony(
            tree = full_tree,
            tip_states = trait,
            check_input = TRUE,
            transition_costs = "proportional",
            edge_exponent = 0,
            weight_by_scenarios = TRUE
        )

        # Extract likelihoods for unknown tips only
        lik <- mp_hsp$likelihoods[unknown_tips_index, , drop=FALSE]
        rownames(lik) <- unknown_tips
        colnames(lik) <- c(0:(ncol(lik)-1))

        # Remove zero-probability columns
        col2remove <- which(colSums(lik) == 0)
        if(length(col2remove) > 0) {
            lik <- lik[, -col2remove, drop=FALSE]
        }

        # Return state with highest probability (subtract 1 to restore original scale)
        as.numeric(colnames(lik)[max.col(lik)])
    })

} else if (hsp_method == "emp_prob") {
    # Add 1 to counts (states must start at 1)
    trait_values <- trait_values + 1

    hsp_results <- lapply(trait_values, function(trait) {
        emp_hsp <- hsp_empirical_probabilities(
            tree = full_tree,
            tip_states = trait,
            check_input = TRUE
        )

        # Extract likelihoods for unknown tips only
        lik <- emp_hsp$likelihoods[unknown_tips_index, , drop=FALSE]
        rownames(lik) <- unknown_tips
        colnames(lik) <- c(0:(ncol(lik)-1))

        # Remove zero-probability columns
        col2remove <- which(colSums(lik) == 0)
        if(length(col2remove) > 0) {
            lik <- lik[, -col2remove, drop=FALSE]
        }

        # Return state with highest probability (subtract 1 to restore original scale)
        as.numeric(colnames(lik)[max.col(lik)])
    })

} else {
    stop(paste("Unknown HSP method:", hsp_method, "(use 'mp' or 'emp_prob')"))
}

# Format output
predicted_values <- data.frame(hsp_results, check.names = FALSE)
predicted_values$sequence <- unknown_tips
predicted_values <- predicted_values[, c("sequence", colnames(trait_values))]

# Write results
write.table(predicted_values, file=output_file, row.names=FALSE, quote=FALSE, sep="\t")
