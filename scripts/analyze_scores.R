library(yogitools)

# indir <- "data/3_scores"
# score_data = read.csv(paste0(indir,"/results.csv"))

indir <- "data"
outdir <- "data/4_analysis"

load_tables <- function(indir) {
  result_files <- list.files(indir,pattern="results.*csv",full.names=TRUE)
  result_tables <- lapply(result_files, read.csv) |>
    lapply(\(tbl) {
      #remove extra columns from pandas
      junk <- grep("^X$|^Unnamed.",colnames(tbl))
      tbl[,-junk]
    })
  do.call(rbind,result_tables)
}
score_data <- load_tables(indir)

#set the quality metrics for unparseable results to 0
score_data[score_data$parsing_quality == "unparseable","f1"] <- 0


#' Prepare a custom legend plotting area
#' 
#' @param where Where the legend should be drawn (topleft, top, right, bottomright, etc)
#' @param w the width of the legend as a fraction of the page width.
#' @param h the height of the legend as a fraction of the page height
#' @param coordinate_limits the x and y limits of the legends user coordinates (default c(0,1,0,1))
#' @param margins the margins of the legend (defaults to c(0,0,0,0))
custom_legend <- function(where="topleft", w=0.4, h=0.3, coordinate_limits=c(0,1,0,1), margins=c(0,0,0,0)) {

  #determine the device coordinates for the edges of the plotting area
  xlimits <- if (par("xlog")) 
    grconvertX(10^par("usr")[1:2],"user","ndc") 
  else 
    grconvertX(par("usr")[1:2],"user","ndc")
  ylimits <- if (par("ylog")) 
    grconvertY(10^par("usr")[3:4],"user","ndc") 
  else 
    grconvertY(par("usr")[3:4],"user","ndc")

  #determine the bounding box of the legend based on those edges,
  # and based on the `where` string 
  fig_box <- switch(
    where,
    topleft = c(
      xlimits[1], xlimits[1]+w, 
      ylimits[2]-h, ylimits[2]
    ),
    top = c(
      mean(xlimits[1:2]-w/2), mean(xlimits[1:2]+w/2), 
      ylimits[2]-h, ylimits[2]
    ),
    topright = c(
      xlimits[2]-w, xlimits[2], 
      ylimits[2]-h, ylimits[2]
    ),
    right = c(
      xlimits[2]-w, xlimits[2], 
      mean(ylimits[1:2])-h/2, mean(ylimits[1:2])+h/2
    ),
    bottomright = c(
      xlimits[2]-w, xlimits[2], 
      ylimits[1], ylimits[1]+h
    ),
    bottom = c(
      mean(xlimits[1:2]-w/2), mean(xlimits[1:2]+w/2), 
      ylimits[1], ylimits[1]+h
    ),
    bottomleft = c(
      xlimits[1], xlimits[1]+w, 
      ylimits[1], ylimits[1]+h
    ),
    left = c(
      xlimits[1], xlimits[1]+w,
      mean(ylimits[1:2])-h/2, mean(ylimits[1:2])+h/2
    )
  )

  #set the graphics parameters
  op <- par(
    fig = fig_box, xlog = FALSE, ylog = FALSE,
    mar = margins, new = TRUE
  )
  #create the plotting area
  plot(
    NA, type="n", 
    xlim=coordinate_limits[1:2],
    ylim=coordinate_limits[3:4],
    axes=FALSE,
    xlab="",
    ylab=""
  )
  #draw a box around the legend
  # do.call(rect,coordinate_limits)
  rect(
    coordinate_limits[1],
    coordinate_limits[3],
    coordinate_limits[2],
    coordinate_limits[4],
    col=yogitools::colAlpha("white",0.5)
  )

  #return the old graphics parameters
  return(op)
}

#' apply a function across category combinations
#' 
#' e.g. multi_tapply(data, "llm", c("prompt","modality"), "f1", mean)
#' would produce a data frame with columns rows corresponding to llm names
#' and columns to prompt-modality conbinations, showing the mean 
#' f1 score for each
#' 
#' @param data input data.frame
#' @param categories1 list of column names to categorize over in the 1st dimension
#' @param categories2 list of column names to categorize over in the 2nd dimension
#' @param operand column name on which the operation is applied
#' @param fun the function to apply across the operand per category combination
#' @param ... additional parameters for the `fun` function
#' @returns A data.frame 
multi_tapply <- function(data, categories1, categories2, operand, fun, ...) {
  #build tags for category 1 combinations
  ctags1 <- apply(data[,categories1, drop=FALSE], 1, paste, collapse="&")
  #use tapply to select subsets
  results <- tapply(seq_len(nrow(data)), ctags1, \(is) {
    #temporarily store the subset
    subset <- data[is,]
    #build tags for category 2 across subset
    ctags2 <- apply(subset[,categories2, drop=FALSE], 1, paste, collapse="&")
    #use tapply to apply
    tapply(subset[,operand], ctags2, fun, ...)
  })
  cols <- Reduce(union,lapply(results, names))
  outmat <- do.call(rbind,lapply(results,`[`,cols))
  colnames(outmat) <- cols
  outmat
}

#' calculate mean, stdev and degrees of freedom across category combinations
#' 
#' @param data input data.frame
#' @param categories1 vector of column names to categorize over in 1st dimension
#' @param categories2 vector of column names to categorize over in 2nd diemension
#' @param operand column name on which to calculate mean, sd and df
multi_msd <- function(data, categories1, categories2, operand) {
  zbind(
    mean=multi_tapply(
      data, categories1, categories2, operand, 
      fun=mean, na.rm=TRUE
    ),
    sd=multi_tapply(
      data, categories1, categories2, operand, 
      fun=sd, na.rm=TRUE
    ),
    df=multi_tapply(
      data, categories1, categories2, operand, 
      fun=\(x) sum(!is.na(x))
    )
  )
}

#############################################
# Pretty-printing names for categorical data
#############################################
tool_names <- c(
  `gpt-4.1-mini` = "GPT4.1 mini",
  `gemma3:27b` = "Gemma3 (27B)",
  `llama3.1:70b` = "Llama3.1 (70B)",
  `mistral-small3.1:latest` = "Mistral (24B)",
  `numind/NuExtract-2.0-2B` = "NuExtract2 (2B)"
)
qual_names = c(
  perfect="Perfect",
  markdown_block="MD-embedded",
  buried="Buried in text",
  repair_needed="Repair needed",
  unparseable="Unparseable"
)
modality_names <- c(
  raw_text="Raw text",
  ocr_text="OCR text",
  image="Image"
)
inqual_names = c(
  original="Original",
  distressed="Faxed"
)
prompt_names = c(
  zero_shot="Zero-shot",
  one_shot="One-shot"
)
template_names <- c(
  cheo = "CHEO",
  hamilton = "HHS",
  sick_kids = "SickKids",
  kingston = "KHSC",
  london = "LHSC",
  nygh = "NYGH",
  mt_sinai = "MtSinai",
  trillium = "Trillium",
  uhn = "UHN",
  fakeHospital1 = "OCH",
  fakeHospital2 = "TCH"
)


########################
# PARSING QUALITY PLOT #
########################

plot_parsing_quality <- function() {
  
  quality_bars = with(
    score_data,
    table(data.frame(
      quality=factor(parsing_quality, levels=names(qual_names)), 
      tool=paste0(tool,"&",modality)
    ))
  ) |> apply(2,\(x)100*x/sum(x))
  plot_colors = c(
    "chartreuse4","chartreuse2","gold2",
    "orange","firebrick3"
  )
  labels = strsplit(colnames(quality_bars),"&") |> 
    sapply(\(fields) paste0(tool_names[fields[[1]]], "\n", modality_names[fields[[2]]]))
  op <- par(mar=c(9,4,1,1),las=2)
  barplot(
    quality_bars,
    names.arg=labels,
    col=plot_colors,
    border=NA,
    ylab="% extractions",
    xlim=c(0,ncol(quality_bars)+6) # add space for the legend
  )
  legend("right",qual_names,fill=plot_colors)
  par(op)
}

pdf(paste0(outdir,"/Fig1A_parsing_quality.pdf"),10,5)
plot_parsing_quality()
dev.off()



base_colors <- c(
  "firebrick","chartreuse","steelblue",
  "darkgoldenrod","darkorchid"
)

tool_colors <- base_colors |> setNames(names(tool_names))
mode_shades <- 2:4 |> setNames(names(modality_names))
parsing_shapes <- setNames(0:4,names(qual_names))


plot_processing_times <- function() {
  plot_colors = with(
    score_data, 
    paste0(tool_colors[tool],mode_shades[modality])
  )

  op <- par(mar=c(5,4,1,1))
  with(score_data,plot(
    output_tokens, 
    processing_time/60,
    xlab="#output tokens",
    ylab="processing time",
    log="xy",
    axes=FALSE,
    pch=parsing_shapes[parsing_quality],
    cex=0.5,
    # col=sapply(plot_colors,yogitools::colAlpha,0.2)
    col=plot_colors
  ))
  axis(1)
  ticks <- c(2/60,5/60,1/6,1/2,1,2,5,10)
  axis(2,ticks,c("2s","5s","10s","30s","1m","2m","5m","10m"))
  abline(h=ticks,col="gray",lty="dashed")
  par(op)

  #create a legend
  op <- custom_legend(w=0.4,h=0.2,coordinate_limits=c(-6,11,0,9))
  #create a matrix of values for the legend
  legend_map <- expand.grid(tool=names(tool_names),modality=names(modality_names))
  legend_map$y <- sapply(legend_map$tool, \(x) which(names(tool_names)==x))
  legend_map$x <- sapply(legend_map$modality, \(x) which(names(modality_names)==x))
  legend_map$shade <- with(legend_map,paste0(tool_colors[tool],mode_shades[modality]))
  #color legend
  with(legend_map, points(x,y,col=shade,pch=20,cex=1))
  # with(legend_map, rect(x-.5,y-.5,x+.5,y+.5,col=shade,border=NA))
  with(legend_map,text(seq_along(modality_names)+.5,max(y)+1,modality_names,pos=3,srt=45,cex=0.5))
  with(legend_map,text(min(x)-.1,seq_along(tool_names),tool_names,pos=2,cex=0.5))
  #shape legend
  x_off <- length(modality_names)+2
  points(rep(x_off,length(parsing_shapes)),seq_along(parsing_shapes),pch=parsing_shapes,cex=0.5)
  text(x_off+.1,seq_along(parsing_shapes),qual_names,pos=4,cex=0.5)
  par(op)

}

pdf(paste0(outdir,"/FigS2_processing_time.pdf"),7,7)
plot_processing_times()
dev.off()


########################
# F1-score distributions
########################

plot_f1_distributions <- function(parseable_only=FALSE) {
  if (parseable_only) {
    filtered_data <- score_data[which(score_data$parsing_quality != "unparseable"),]
  } else {
    filtered_data <- score_data
  }
  #Calculate the histogram data across categories
  categories <- c("tool", "modality", "quality")
  category_tags <- apply(filtered_data[,categories, drop=FALSE], 1, paste, collapse="&")
  breaks <- seq(0,100)
  histos <- with(filtered_data, tapply(f1*100, category_tags, hist, breaks=breaks, plot=FALSE))
  
  #figure out the category labels belonging to each histogram
  labels <- strsplit(names(histos),"&")|>as.df()
  colnames(labels) <- categories

  #where to draw the labels for the tools
  llm_label_pos <- tapply(seq_len(nrow(labels))-.5, labels$tool, mean)
  #where to draw the labels for the modalities
  modality_label_pos <- tapply(seq_len(nrow(labels)), labels$tool, \(is) {
    tapply(is-.5, labels[is, "modality"], mean)
  }) |> setNames(NULL) |> unlist()
  #where to draw the labels for qualities
  qual_label_pos <- (seq_len(nrow(labels))-.5) |> setNames(labels$quality)
  #where to draw the dividers between tool secdtions
  llm_dividers <- tapply(seq_len(nrow(labels)), labels$tool, max)

  #heights, x and y positions of bars in barplots
  hs <- do.call(rbind,lapply(histos, \(h) h$counts / max(h$counts)))
  xs <- t(replicate(length(histos),seq(1,100)))
  ys <- replicate(100, seq_along(histos)-1)
  #colors of the bars
  qual_colors = c(image="steelblue",ocr_text="chartreuse",raw_text="firebrick")
  bar_colors <- mapply(\(col,shade) paste0(col,shade),
    col=qual_colors[labels[,2]], 
    # col=sapply(labels[,2]=="image",ifelse,"steelblue","firebrick"), 
    shade=sapply(labels[,3]=="original",ifelse,2,4)
  )

  #set the margins
  op <- par(las=1, mar=c(5,1,1,1))
  #create the plotting area
  plot(
    NA, type="n", 
    xlab=expression(F[1]), ylab="",
    xlim=c(-70,101), ylim=c(0, length(histos)),
    axes=FALSE
  )
  #draw the histogram bars
  rect(xs-1, ys, xs, ys+hs, col=bar_colors[ys+1], border=NA)
  #draw the text labels
  text(-40, llm_label_pos, tool_names[names(llm_label_pos)], pos=2)
  text(-20, modality_label_pos, modality_names[names(modality_label_pos)], pos=2)
  text(-1, qual_label_pos, inqual_names[names(qual_label_pos)], pos=2)
  #draw the dividers
  abline(h = c(0,llm_dividers), col="gray")
  abline(v = seq(0,100,20), col="gray", lty="dashed")
  axis(1, seq(0, 100, 20))
  #reset graphics parameters
  par(op)
}

pdf(paste0(outdir,"/Fig1D_f1_distributions.pdf"),7,7)
plot_f1_distributions(parseable_only=TRUE)
dev.off()

pdf(paste0(outdir,"/FigS1D_f1_distributions_all.pdf"),7,7)
plot_f1_distributions(parseable_only=FALSE)
dev.off()

#####################
# Statistical tests between distributions
#####################

list(
  list(
    question="Is Gemma better than GPT on raw text?",
    p=with(score_data,wilcox.test(
      f1[which(tool=="gemma3:27b" & prompt=="zero_shot" & modality=="raw_text")],
      f1[which(tool=="gpt-4.1-mini" & prompt=="zero_shot" & modality=="raw_text")],
      paired=TRUE,alternative="greater"
    ))$p.value
  ),
  list(
    question="Is GPT better than Gemma on Images?",
    p=with(score_data,wilcox.test(
      f1[which(tool=="gemma3:27b" & prompt=="zero_shot" & modality=="image" & quality=="original")],
      f1[which(tool=="gpt-4.1-mini" & prompt=="zero_shot" & modality=="image"& quality=="original")],
      paired=TRUE,alternative="less"
    ))$p.value
  ),
  list(
    question="Are zero-shot prompts better than one-shot on gemma/raw?",
    p=with(score_data,wilcox.test(
      f1[which(tool=="gemma3:27b" & prompt=="zero_shot" & modality=="raw_text")],
      f1[which(tool=="gemma3:27b" & prompt=="one_shot" & modality=="raw_text")],
      paired=TRUE,alternative="greater"
    ))$p.value
  ),
  list(
    question="Is GPT better on images than on OCR?",
    p=with(score_data,wilcox.test(
      f1[which(tool=="gpt-4.1-mini" & prompt=="zero_shot" & modality=="image" & quality=="original")],
      f1[which(tool=="gpt-4.1-mini" & prompt=="zero_shot" & modality=="ocr_text" & quality=="original")],
      paired=TRUE,alternative="greater"
    ))$p.value
  ),
  list(
    question="Is Gemma better on OCR than on images?",
    p=with(score_data,wilcox.test(
      f1[which(tool=="gemma3:27b" & prompt=="zero_shot" & modality=="image" & quality=="original")],
      f1[which(tool=="gemma3:27b" & prompt=="zero_shot" & modality=="ocr_text" & quality=="original")],
      paired=TRUE,alternative="less"
    ))$p.value
  ),
  list(
    question="Is GPT better with one-shot prompts?",
    p=with(score_data,wilcox.test(
      f1[which(tool=="gpt-4.1-mini" & prompt=="zero_shot" & modality=="raw_text" & quality=="original")],
      f1[which(tool=="gpt-4.1-mini" & prompt=="one_shot" & modality=="raw_text" & quality=="original")],
      paired=TRUE,alternative="less"
    ))$p.value
  )
) |> as.df() -> pvals
pvals$q <- p.adjust(pvals$p, method="fdr")


# filtered_data <- score_data[which(score_data$quality=="original"),]
# f1_by_tag <- with(filtered_data,tapply(f1, paste(tool,prompt,modality,sep="&") ,c))
# cats <- do.call(rbind,strsplit(names(f1_by_tag),"&"))

# #generate all testing combinations
# all_tests <- combn(names(f1_by_tag),2,simplify=FALSE) |> lapply(\(x) list(
#   a=x[[1]],
#   b=x[[2]],
#   p=wilcox.test(f1_by_tag[[x[[1]]]],f1_by_tag[[x[[2]]]],paired=TRUE)$p.value
# ))|>as.df()
# all_tests$q <- p.adjust(all_tests$p,method="fdr")
# all_tests[all_tests$q < 0.05,1:2]


####################
# Correlation across documents
####################
plot_doc_cor <- function() {
  filtered_data <- score_data[with(score_data,which(quality=="original" & modality=="ocr_text" & prompt=="zero_shot")),]
  docs_vs_tools <- do.call(rbind,with(filtered_data,tapply(seq_len(nrow(filtered_data)),doc_id,\(is){
    f1[is]|>setNames(tool[is])
  })))
  pairs(
    docs_vs_tools[,names(tool_names)],
    labels=tool_names,
    xlim=c(0,1),
    ylim=c(0,1),
    pch=20,
    col=yogitools::colAlpha("steelblue3",0.5),
    cex.labels=1,
    upper.panel=\(x,y,...) {
      abline(0,1,col="gray")
      points(x,y,...)
    },
    lower.panel=\(x,y,...) {
      op <- par(usr=c(0,1,0,1))
      r2 <- round(cor(x,y)^2,digits=2)
      text(0.5, 0.5, bquote(R^2 == .(r2)))
      par(op)
    }
  )
}

pdf(paste0(outdir,"/FigS4_document_correlation.pdf"),7,7)
plot_doc_cor()
dev.off()



#####################
# F1 score means barplot
#####################

plot_f1_means_by_prompt <- function(distressed=FALSE) {
  if (distressed) {
    filtered_data <- score_data[which(score_data$quality == "distressed"),]
    prompt_mod_order <- names(prompt_names) |> lapply(\(x)paste0(x,"&",names(modality_names[-1]))) |> unlist()
  } else {
    filtered_data <- score_data[which(score_data$quality == "original"),]
    prompt_mod_order <- names(prompt_names) |> lapply(\(x)paste0(x,"&",names(modality_names))) |> unlist()
  }

  msd <- multi_msd(
    filtered_data,
    categories1 = c("tool"),
    categories2 = c("prompt", "modality"),
    operand = "f1"
  )

  # #sort by label order above
  msd <- msd[names(tool_names), prompt_mod_order, ]

  prompt_colors <- c("steelblue","darkorange") |> setNames(names(prompt_names))
  mode_shades <- 2:4 |> setNames(names(modality_names))

  bar_colors <- colnames(msd) |> strsplit("&") |> sapply(\(fields)paste0(prompt_colors[fields[[1]]],mode_shades[fields[[2]]])) |> setNames(colnames(msd))

  qual_label <- if (distressed) "(fax quality)" else "(original quality)"

  # set margins and axis label orientations
  op <- par(las=3, mar=c(8,4,4,1))
  # draw plot
  xs <- barplot(
      t(msd[,,1]), beside=TRUE, space=c(0,1.5),
      names.arg=tool_names,
      ylim=c(0,1),
      col=bar_colors, border=NA, 
      ylab=expression("Mean"~F[1]~"score"), 
      main=bquote("Mean"~F[1]~.(qual_label))
  )
  stderr <- msd[,,2]/sqrt(msd[,,3])
  # draw the error bars
  errorBars(
    xs, t(msd[,,1]), t(stderr), 
    l=.04, col="gray30"
  )
  # determine x-coordinates of missing values
  xna <- xs[apply(t(msd[,,1]),1:2,is.na)]
  # add hatching at missing value positions
  rect(xna-0.5, 0, xna+0.5, 100, col="gray", density=20,border=NA)
  #draw stat-test brackets
  if (!distressed){
    drawPvalBracket(pvals$q[[1]], xs[1,1], xs[1,2], h = .7, th = .02)
    drawPvalBracket(pvals$q[[2]], xs[3,1], xs[3,2], h = .8, th = .02)
    drawPvalBracket(pvals$q[[3]], xs[1,2], xs[4,2], h = .7, th = .02)
    drawPvalBracket(pvals$q[[4]], xs[2,1], xs[3,1], h = .6, th = .02)
    drawPvalBracket(pvals$q[[5]], xs[2,2], xs[3,2], h = .6, th = .02)
  }
  # drawPvalBracket(pvals[[1]], xs[2,1], xs[2,2], h = 88, th = 2)
  # drawPvalBracket(pvals[[2]], xs[1,1], xs[3,1], h = 78, th = 2)
  # add grid lines
  grid(NA,NULL)
  # add legend
  # legend("topright", colnames(msd), fill=bar_colors, bg="white")
  op <- custom_legend("topright",w=0.3,h=0.2,coordinate_limits=c(-.1,7,0,5))
  #create a matrix of values for the legend
  legend_map <- expand.grid(prompt=names(prompt_names),modality=names(modality_names))
  legend_map$y <- sapply(legend_map$prompt, \(x) which(names(prompt_names)==x))
  legend_map$x <- sapply(legend_map$modality, \(x) which(names(modality_names)==x))
  legend_map$shade <- with(legend_map,paste0(prompt_colors[prompt],mode_shades[modality]))
  #color legend
  # with(legend_map, points(x,y,col=shade,pch=20,cex=1))
  with(legend_map, rect(x-.5,y-.5,x+.5,y+.5,col=shade,border=NA))
  with(legend_map,text(seq_along(modality_names)+.8,max(y)+1,modality_names,pos=3,srt=45,cex=1))
  with(legend_map,text(max(x)+.6,seq_along(prompt_names),prompt_names,pos=4,cex=1))
  par(op)
}

pdf(paste0(outdir,"/Fig1B_f1_means.pdf"),7,7)
plot_f1_means_by_prompt(distressed=FALSE)
dev.off()

pdf(paste0(outdir,"/FigS1B_f1_means_distressed.pdf"),7,7)
plot_f1_means_by_prompt(distressed=TRUE)
dev.off()


###################
# F1 by template
##################

plot_f1_by_template <- function() {

  get_tool_mod_names <- function(tags) {
    strsplit(tags,"&") |>
      sapply(\(fields) paste0(
        tool_names[fields[[1]]], "\n", 
        modality_names[fields[[2]]]
      ))
  }

  f1_by_template <- multi_tapply(
    score_data, "template", c("tool", "modality"), "f1", 
    fun=mean, na.rm=TRUE
  )

  tool_means <- apply(f1_by_template, 2, mean)
  template_means <- apply(f1_by_template, 1, mean)

  #hierarchical clustering of templates by similarity
  distmat <- dist(f1_by_template)
  clusters <- hclust(distmat)

  #sort rows and columns by mean_f1 and clustering order
  tool_order <- order(tool_means, decreasing=TRUE)
  template_order <- clusters$order
  f1 <- as.matrix(f1_by_template[template_order, tool_order])
  tool_means <- tool_means[tool_order]
  template_means <- template_means[template_order]

  cmap <- yogitools::colmap(
    c(min(f1),median(f1),max(f1)), 
    c("royalblue3", "white", "firebrick3")
  )


  #set layout: main plot in bottom left, 
  #  histograms above and to the right
  layout(
    rbind(
      c(2, 4),
      c(1, 3)
    ), 
    widths = c(8, 2),
    heights = c(2, 8)
  )

  bmarg <- 10
  #set axis labels to perpendicular mode, and set custom margins
  op <- par(las=2, mar=c(bmarg,5,0,0)+.1)
  plot(NA, 
    xlim=c(0.5,ncol(f1)+.5), ylim=c(0.5,nrow(f1)+.5), 
    type="n", axes=FALSE,
    xlab="", ylab=""
  )
  axis(1,1:ncol(f1),get_tool_mod_names(colnames(f1)))
  axis(2,nrow(f1):1, template_names[rownames(f1)])
  xs <- do.call(c,lapply(1:ncol(f1), \(i) rep(i,nrow(f1))))
  ys <- rep(seq(nrow(f1),1),ncol(f1))
  rect(xs-.5,ys-.5,xs+.5, ys+.5, col=apply(f1,1:2,cmap), border=NA)
  text(xs,ys,sprintf("%.3f",f1),cex=.7)
  par(op)

  #histogram on the top
  op <- par(mar=c(0,5,1,0)+.1)
  barplot(tool_means, 
    # col = sapply(tool_means, cmap),
    names.arg="", ylab=expression(E(F[1])),
    space=0, axes=FALSE, border=NA,
  )
  axis(2)
  par(op)

  #histogram on the right
  op <- par(mar=c(bmarg,0,0,1)+.1)
  barplot(rev(template_means), 
    # col = sapply(rev(template_means), cmap),
    names.arg="", xlab=expression(E(F[1])),
    horiz=TRUE, space=0, axes=FALSE, border=NA
  )
  axis(1)
  par(op)

  #legend
  op <- par(mar=c(5,0.1,2,0.1))
  plot(NA,type="n",xlim=c(0,1),ylim=c(0,1),axes=FALSE,xlab=expression(F[1]))
  step <- 0.01
  x <- seq(0,1,step)
  rect(x-step/2,0,x+step/2,1,col=cmap(x),border=NA)
  axis(1)
  par(op)
}

pdf(paste0(outdir,"/Fig3_templates.pdf"),7,7)
plot_f1_by_template()
dev.off()


#############################
# Prompt difference cards
###########################


#' calculate differences and MWU tests between two ub-populations
#' @param subpops list lists of f1 scores (first split by prompt, then by tool/modality)
compare_subpopulations <- function(subpops) {
  #which tool/modality combos occur for both prompts (and can thus be tested)?
  tool_and_mod_combos <- Reduce(intersect,lapply(subpops,names))
  subpops <- lapply(subpops, \(sp) sp[tool_and_mod_combos])
  #for each combo, compare run wilcox tests 
  out <- lapply(tool_and_mod_combos, \(sp_name) {
    m1 <- mean(subpops[[1]][[sp_name]],na.rm=TRUE)
    m2 <- mean(subpops[[2]][[sp_name]],na.rm=TRUE)
    d <- m2-m1
    p <- wilcox.test(
      subpops[[1]][[sp_name]], 
      subpops[[2]][[sp_name]],
      paired=TRUE
    )$p.value
    c(m1=m1, m2=m2, d=d, p=p)
  }) |> setNames(tool_and_mod_combos) |> as.df()
  out$q <- p.adjust(out$p,method="fdr")
  out
}

# parsed_data <- data[data$Parsed,]

modalities <- c("ocr_text","image")
#compare by prompt
subpops <- lapply(names(prompt_names), \(prompt) {
  subset <- score_data[which(score_data$prompt==prompt & score_data$quality=="original" & score_data$modality %in% modalities),]
  tags <- apply(subset[,c("tool","modality"), drop=FALSE], 1, paste, collapse="&")
  tapply(subset$f1, tags, c)
}) |> setNames(names(prompt_names))
changes_by_prompt <- compare_subpopulations(subpops)

#compare by input
# modalities <- c("ocr_text","image")
subpops <- lapply(modalities, \(mod) {
  subset <- score_data[which(score_data$modality==mod & score_data$quality=="original"),]
  tags <- apply(subset[,c("tool","prompt"), drop=FALSE], 1, paste, collapse="&")
  tapply(subset$f1, tags, c)
}) |> setNames(modalities)
changes_by_modality <- compare_subpopulations(subpops)

make_text <- function(changes,catnames) {
  split_labels <- strsplit(rownames(changes),"&")
  #helper function to generate significance stars based on p
  # pstars <- function(p) {
  #   c("n.s.","*","**","***")[sum(p < c(Inf,0.05,0.01,0.001))]
  # }
  # qExpr <- function(p) {
  #   if (p < 0.001) {
  #     if (p < 2.2e-16) {
  #       expression(q < 2.2 %*% 10^-16)
  #     } else {
  #       expo <- floor(log10(p))
  #       sfd <- signif(p * 10^-expo, digits = 3)
  #       bquote(q == .(sfd) %*% 10^.(expo))
  #     }
  #   } else {
  #     sprintf("q = %.03f", p)
  #   }
  # }
  lapply(seq_len(nrow(changes)), \(i) {
    sprintf(
      "%s (%s)\n%s%.1f (%.1f->%.1f)",
      tool_names[split_labels[[i]][[1]]],
      catnames[split_labels[[i]][[2]]], 
      ifelse(changes[i,"d"]>0,"+",""),
      changes[i,"d"]*100, 
      changes[i,"m1"]*100, 
      changes[i,"m2"]*100#,
      # changes[i,"q"]
    )
  })
}
# make_expressions <- function(changes) {
qExpr <- function(p) {
  if (p < 0.001) {
    if (p < 2.2e-16) {
      expression(q < 2.2 %*% 10^-16)
    } else {
      expo <- floor(log10(p))
      sfd <- signif(p * 10^-expo, digits = 3)
      bquote(q == .(sfd) %*% 10^.(expo))
    }
  } else {
    sprintf("q = %.03f", p)
  }
}
#   lapply(changes[,"q"],qExpr)
# }

drawChangePlot <- function(changes, xoff=0, cmap=colmap(c(-.5,0,.5)), title="", catnames=modality_names) {
  changes <- changes[order(changes$d),]
  rect_colors <- rowApply(changes, \(d, p, ...) ifelse(p < 0.05, cmap(d), "white")) |> unlist()
  ys <- seq_len(nrow(changes))
  rect(xoff,ys-1,xoff+1,ys,col=rect_colors,border=NA)
  text(xoff+.5,ys-.5,make_text(changes, catnames),cex=0.6)
  # text(xoff+.5,ys-.75,make_expressions(changes),cex=0.6)
  # exprs <- make_expressions(changes)
  for (i in seq_len(nrow(changes))) {
    text(xoff+.5,ys[[i]]-.85,qExpr(changes[i,"q"]),cex=0.6)
  }
  text(xoff+.5,nrow(changes)+1, title,cex=0.7)
}

pdf(paste0(outdir,"/Fig1C_change_panels.pdf"),5,5)
op <- par(mar=c(.1,.1,.1,.1))
plot(
  NA,type="n",
  xlim=c(0,2.5),ylim=c(0,10),
  axes=FALSE, xlab="", ylab=""
)
drawChangePlot(changes_by_prompt, 
  title="Prompting impact\nZero-shot->One-shot\nDifference in F1",
  catnames=modality_names
)
drawChangePlot(changes_by_modality, xoff=1.2, 
  title="Modality impact\nText->Image\nDifference in F1",
  catnames=prompt_names
)
par(op)
dev.off()


################################
# Precision/recall by field name
################################


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

### Load the field-specific data
pr_rec_all <- array(
  NA,
  dim=c(length(field_names),length(tool_names),2),
  dimnames=list(field_names,names(tool_names),c("precision","recall"))
)
infiles <- list.files("data",pattern="pr_rec_by_field_",full.names=TRUE) 
for (infile in infiles) {
  load(infile)
  #both ifelse and is.na operate element-wise
  pr_rec_all <- ifelse(is.na(pr_rec), pr_rec_all, pr_rec)
}


plot_field_metrics <- function() {

  cmap <- colmap(
    c(0, .5, 1),
    c("steelblue3", "white", "darkorange3")
  )
  y <- rep(rev(seq_len(nrow(pr_rec_all))),ncol(pr_rec_all))
  x <- do.call(c,lapply(seq_len(ncol(pr_rec_all)),\(i)rep(i,nrow(pr_rec_all))))

  op <- par(las=2,mar=c(10,10,1,1),mfrow=c(1,3))

  plot(
    NA, type="n",
    xlim=c(0,ncol(pr_rec_all)+1), 
    ylim=c(0,nrow(pr_rec_all)+1),
    xlab="", ylab="", axes=FALSE,
    main="Recall"
  )
  axis(1, seq_len(ncol(pr_rec_all)), tool_names[colnames(pr_rec_all)])
  axis(2, rev(seq_len(nrow(pr_rec_all))), rownames(pr_rec_all))
  rect(x-.5,y-.5,x+.5,y+.5,border=NA,col=cmap(pr_rec_all[,,"recall"]))

  plot(
    NA, type="n",
    xlim=c(0,ncol(pr_rec_all)+1), 
    ylim=c(0,nrow(pr_rec_all)+1),
    xlab="", ylab="", axes=FALSE,
    main="Precision"
  )
  axis(1, seq_len(ncol(pr_rec_all)), tool_names[colnames(pr_rec_all)])
  axis(2, rev(seq_len(nrow(pr_rec_all))), rownames(pr_rec_all))
  rect(x-.5,y-.5,x+.5,y+.5,border=NA,col=cmap(pr_rec_all[,,"precision"]))

  par(mar=c(10,5,10,10))

  plot(NA,type="n",xlim=c(0,1),ylim=c(0,1),axes=FALSE,ylab="Value",xlab="")
  step <- 0.01
  y <- seq(0,1,step)
  rect(0,y-step/2,1,y+step/2,col=cmap(y),border=NA)
  axis(2)

  par(op)

}

pdf(paste0(outdir,"/FigS3_field_metrics.pdf"),7,7)
plot_field_metrics()
dev.off()
