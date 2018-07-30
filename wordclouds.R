
options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE)

topic.wordl <- function(filepath, wordsize_scale=0.75){
  library("tm")
  library("SnowballC")
  library("wordcloud")
  library("RColorBrewer")
  
 Encoding('UTF-8') 
  
  
  text <- readLines(filepath, encoding="ASCII")
  #Encoding(text)  <- "UTF-8"
  #print(text)
  # Load the data as a corpus
  docs <- Corpus(VectorSource(text), readerControl = list(language = "german"))
  
  toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
  docs <- tm_map(docs, toSpace, "/")
  docs <- tm_map(docs, toSpace, "@")
  docs <- tm_map(docs, toSpace, "\\|")

#   Convert the text to lower case
  docs <- tm_map(docs, content_transformer(tolower))
  
#   Remove numbers
  docs <- tm_map(docs, removeNumbers)
  
#   Remove english common stopwords
  docs <- tm_map(docs, removeWords, stopwords("english"))
  docs <- tm_map(docs, removeWords, stopwords("german"))
  
   #Remove punctuations
  docs <- tm_map(docs, removePunctuation)
   #Eliminate extra white spaces
  docs <- tm_map(docs, stripWhitespace)
  

  pal2 <- brewer.pal(8,"Dark2")
  wordcloud(docs, max.words = 100,  min.freq=1, random.order=FALSE, scale=c(2,.2), colors=pal2)
}

files <- list.files(path=sprintf(".\\%s\\wordclouds", args), pattern='*.txt')
# files

#pdf(file=".\\Output\\wordclouds\\wordcloud.pdf")
for (file in files){
  # pdf(file=".\\Output\\wordclouds\\wordcloud.pdf")
  pdf(sub(".txt", ".pdf", sprintf(".\\%s\\wordclouds\\%s", args, file)))
  # pdf(sprintf(".\\Output\\wordclouds\\%s.pdf", file))
  
  # svg(sprintf(".\\Output\\wordclouds\\%s.svg", file))
  
  fullpath = sprintf(".\\%s\\wordclouds\\%s", args, file)
  topic.wordl(fullpath, wordsize_scale = 0.5)
  dev.off()
	}
#dev.off()
