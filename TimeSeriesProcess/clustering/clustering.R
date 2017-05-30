# Joaquim Leitão - 2011150072
# 2016/2017 School Year
# Doctoral Program in Information Science and Technology - Connectivity and Pattern Recognition
# Project

library(dtwclust)
library(dtw)

loadImputedFile <- function(imputedDataPath) {
  # Load imputed file
  data <- read.csv2(imputedDataPath, header=FALSE, sep=";")
  dataNames <- names(data)
  
  time <- data[dataNames[1]]
  readings <- data[dataNames[2:length(dataNames)]]
  
  time <- as.matrix(time, byrow=TRUE)
  readings <- as.matrix(readings, byrow=TRUE)
  
  return(list('time'=time, 'readings'=readings))
}

zNormalise <- function(readings) {
  dimensions <- dim(readings)
  nrows <- dimensions[1]
  ncols <- dimensions[2]
  
  for (j in 1:ncols) {
    currVarMean <- mean(as.numeric(readings[1:nrows, j]))
    currVarStd <- sd(as.numeric(readings[1:nrows, j]))
    
    for (i in 1:nrows) {
      readings[i, j] <- (as.numeric(readings[i, j]) - currVarMean)/currVarStd
    }
  }
  
  # Convert to matrix
  readings <- mapply(readings, FUN=as.numeric)
  return(matrix(data=readings, ncol=ncols, nrow=nrows))
}

computeSSE <- function(readings_znormalise, assignments, centroids) {
  assignmentsFactor <- as.factor(assignments)
  assignmentsLevels <- levels(assignmentsFactor)
  sseSum <- 0
  
  dimensions <- dim(readings_znormalise)
  nrows <- dimensions[1]
  ncols <- dimensions[2]
  
  for (key in assignmentsLevels) {
    currCentroid <- centroids[[as.integer(key)]]

    for (j in 1:nrows) {
      if (assignments[j] == key) {
        # Compute distance only for elements in the current cluster (Distance to the centroid)
        currDist <- dtw(currCentroid, readings_znormalise[j, 1:ncols])
        currDist <- currDist$distance
        
        sseSum <- sseSum + (currDist * currDist)
      }
    }
  }
  
  return(sseSum)
}

tadCluster <- function(readings_znormalise) {
  # Implement TADPole Clustering
  dimensions <- dim(readings_znormalise)
  nrows <- dimensions[1]
  ncols <- dimensions[2]
  
  filePath <- '/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/metrics_results.csv'
  
  for (curr_k in 2:8) {
    print(paste('===================== K = ', curr_k, ' =======================================', sep=''))
    clusterResult <- tsclust(readings_znormalise, type="tadpole", k=curr_k, trace=TRUE,
                             control=tadpole_control(dc=1.5, window.size=23))
    # plot(clusterResult, clus=1:curr_k)
    
    # Compute metrics
    sseSum <- computeSSE(readings_znormalise, clusterResult@cluster, clusterResult@centroids)
    avSseSum <- sseSum / nrows
    cviResult <- cvi(clusterResult, type='internal')

    print(sseSum)
    print(cviResult)
    
    # Append to csv
    dataWrite <- matrix(c('raw', curr_k, 'dtw', 'TADPole', cviResult['Sil'], cviResult['CH'], sseSum, avSseSum),
                        nrow=1, ncol=8)
    write.table(dataWrite, file=filePath, row.names=FALSE, col.names=FALSE, sep=';', append=TRUE, quote=FALSE)
    
    # Further Cluster evaluation (only for k=3 and k=4)
    # TODO(jpleitao)
  }
}
