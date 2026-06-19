
indir <- "."
outdir <- "joint"
dir.create(outdir)

result_files <- list.files(indir,pattern="results.*csv",full.names=TRUE)
result_tags <- do.call(rbind,strsplit(result_files,"_|\\."))[,3]
result_tables <- lapply(result_files, read.csv) |>
  lapply(\(tbl) {
    #remove extra columns from pandas
    junk <- grep("^X$|^Unnamed.",colnames(tbl))
    tbl[,-junk]
  }) |> 
  setNames(result_tags)

for (tag in result_tags) {
  # result_tables[[tag]]$response <- paste0(tag,"/",result_tables[[tag]]$response)
  result_tables[[tag]]$response <- sub("data",tag,result_tables[[tag]]$response)
  # result_tables[[tag]]$scores <- paste0(tag,"/",result_tables[[tag]]$scores)
  result_tables[[tag]]$scores <- sub("data",tag,result_tables[[tag]]$scores)
  result_tables[[tag]]$tag <- tag
}

score_data <-do.call(rbind,result_tables)

#set the quality metrics for unparseable results to 0
score_data[score_data$parsing_quality == "unparseable","f1"] <- 0

#re-assign non-duplicated run ids
joint_ids <- nrow(score_data) |> seq_len() |> sapply(\(n) sprintf("EXP%06d",n))
score_data$run_id <- joint_ids

#move detail files to joint folder
for (i in seq_len(nrow(score_data))) {
  file.copy(score_data[i,"response"],paste0(outdir,"/",joint_ids[[i]],"_response.json"))
  file.copy(score_data[i,"scores"],paste0(outdir,"/",joint_ids[[i]],"_scores.csv"))
}

#remove file links
score_data$response <- NULL
score_data$scores <- NULL
#export for zenodo
write.csv(score_data, paste0(outdir,"/results.csv"),row.names=FALSE)

