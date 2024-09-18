#!/usr/bin/env Rscript


# install.packages('Seurat')
# install.packages('st')
# install.packages('tools')
args = commandArgs(trailingOnly=TRUE)
require('st')
require('Seurat')
library('tools')
options(future.globals.maxSize = 8000 * 1024^2)
# test if there is at least one argument: if not, return an error
if (length(args)==0) {
	  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else if (length(args)==1) {
	  # default output file
	  args[2] = "out.txt"
}

print(args[0])
print(args[1])
print(args[2])


input_filename = args[1]
output_filename = args[2]

fileext = file_ext(input_filename)
print(fileext)

if (fileext=='h5') {
	st.matrix.data <- Seurat::Read10X_h5(input_filename)
} else {
 st.matrix.data <- as.matrix(read.table(paste0(input_filename),check.names=F))
}

colnames(st.matrix.data) <- gsub("X","",colnames(st.matrix.data))

set.seed(123456)
st.matrix.data.vst <- sctransform::vst(st.matrix.data, min_cells=5)$y
                        st.matrix.data.vst <- round(st.matrix.data.vst,3)

write.table(st.matrix.data.vst, paste0(output_filename), quote=F, sep="\t")





