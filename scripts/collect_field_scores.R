# library(yogitools)
indir <- "data/3_scores"

experiments <- read.csv(paste0(indir,"/results.csv"))

tool_names <- c(
  `gpt-4.1-mini` = "GPT4.1 mini",
  `gemma3:27b` = "Gemma3 (27B)",
  `llama3.1:70b` = "Llama3.1 (170B)",
  `mistral-small3.1:latest` = "Mistral (24B)",
  `numind/NuExtract-2.0-2B` = "NuExtract2 (2B)"
)

field_names <- c(
  "date_collected",
  "date_received",
  "date_verified",
  "report_type",
  "testing_context",
  "ordering_clinic",
  "testing_laboratory",
  "sequencing_scope",
  "tested_genes",
  "gene_symbol",
  "refseq_mrna",
  "num_tested_genes",
  "sample_type",
  "analysis_type",
  "reference_genome",
  "variants",
  "variant_id",
  "chromosome",
  "hgvsg",
  "hgvsc",
  "hgvsp",
  "transcript_id",
  "exon",
  "zygosity",
  "interpretation",
  "maf",
  "type"
)

metrics <- c("tp","fp","fn")

counts <- array(
  0,
  dim=c(length(field_names),length(tool_names),length(metrics)),
  dimnames=list(field_names,names(tool_names),metrics)
)


for (i in seq_len(nrow(experiments))) {
  tool <- experiments[i,"tool"]
  score_file <- experiments[i,"scores"]
  if (score_file != "<failed>") {
    scores <- read.csv(score_file)
    increments <- sapply(field_names, \(name) {
      scores[grep(name,scores$ref),metrics] |> colSums()
    }) |> t()
    counts[,tool, ] <- counts[,tool,] + increments
  }
}


pr_rec <- array(
  NA,
  dim=c(length(field_names),length(tool_names),2),
  dimnames=list(field_names,names(tool_names),c("precision","recall"))
)
pr_rec[,,"precision"] <- counts[,,"tp"] / (counts[,,"tp"]+counts[,,"fp"])
pr_rec[,,"recall"] <- counts[,,"tp"] / (counts[,,"tp"]+counts[,,"fn"])


save(pr_rec,file="data/pr_rec_by_field.Rdata")
q()

#### combine the data from different machines

pr_rec_all <- array(
  NA,
  dim=c(length(field_names),length(tool_names),2),
  dimnames=list(field_names,names(tool_names),c("precision","recall"))
)
infiles <- list.files("data",pattern="pr_rec_by_field_",full.names=TRUE) 
for (infile in infiles) {
  load(infile)
  pr_rec_all <- ifelse(is.na(pr_rec), pr_rec_all, pr_rec)
}
