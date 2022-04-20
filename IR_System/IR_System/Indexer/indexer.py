from fileinput import filename
from bs4 import BeautifulSoup # Beautiful Soup is a Python library for extracting data from HTML and XML files
import os
import re
import math
import pickle


path = r'C:\Users\Chris Hunter\Desktop\Info_Retrieval_Final_Project\html_files'


def InverseDF(term, corpus_count, index):

# Inverse document frequency (IDF) auxiliary function; uses the inverted index to determine the IDF for a term and corpus

# idf(t) = log(N/(df + 1)); N = total number of files in corpus; df(t) = number of files in corpus that contain the term t

  df = len(index[term])
  
  return math.log10(corpus_count/(df + 1))  # smoothing the score to avoid division by zero via logbase 10


def TermFrequency(term, file):

# Term frequency (TF) auxiliary function; determines the TF of a word in an html file

# tf(t,d) = count of t in d / number of words in d

  termCount = 0       # Final count of the number of times the given term appear in file

  wordCount = 0       # Total number of words in file

  for word in file:

    if word == term:
      
      termCount += 1

    wordCount += 1

  return (termCount/wordCount)

def TF_IDFIndex(valueType):  # valueType indicates the value structure of dictionary; if it's 0 use filenames; if it's 1 use docIDs

  invertedIndex = {} # dictionary to be returned
  filenames ={} # dictionary to ensure filenames are used instead of vague docIDs
  corpus = {} # dictionary with html filename as the key and the contents of the html value as the value
  docID = 0
  
  for file in os.listdir(path): # os.listdir returns a list of the all html files on my local disk
    
    docID+=1

    with open(os.path.join(path, file), encoding="utf8") as html_file:
      
      soup = BeautifulSoup(html_file.read(), 'lxml')
      
      processedTxt = []
      
      for tag in soup.find_all('p'):
        
        processedTxt.extend((re.sub('[^A-Za-z0-9]+',' ',' '.join(tag.text.split())).split()))

      for tag in soup.find_all('dd'):

          processedTxt.extend((re.sub('[^A-Za-z0-9]+',' ',' '.join(tag.text.split())).split()))

      corpus[file] = processedTxt

      html_file.close()

      for word in processedTxt:

         if word in invertedIndex and docID not in invertedIndex[word]:

           invertedIndex[word].append(docID)

         else:

           invertedIndex[word] = [docID]

      filenames[docID] = file

  for key in invertedIndex.keys():
        
        docIDLst = invertedIndex[key]
        
        idf_score = InverseDF(key, len(os.listdir(path)), invertedIndex)
        
        invertedIndex[key] = []
        
        for docID in docIDLst:
          
          tf_score = TermFrequency(key,corpus[file])

          if valueType != 1:
            
            invertedIndex[key].append((filenames[docID],(tf_score*idf_score)))

          else:

            invertedIndex[key].append((docID,(tf_score*idf_score)))
   
  return invertedIndex


def QueryVector(query, tf_idf_index):

# Query vector function

  qvDict = {}           # Query vector dictionary; will be the final return value

  tokenizedQuery = list(query.split(" "))

  for token in tokenizedQuery:

    if token not in tf_idf_index:

      qvDict[token] = 0
      continue

    qvDict[token] = InverseDF(token, len(os.listdir(path)), tf_idf_index)

  return qvDict


def DocLength(queryVector, docID, tf_idf_index):

# Document length auxiliary function; returns the euclidean length of a document

  sum = 0

  for term in queryVector:

    if docID not in tf_idf_index[term]:

      continue

    for pair in tf_idf_index[term]:

      if pair[0] == docID:

        sum += pair[1] ** 2

        break
  
  return math.sqrt(sum)


def CSSearch(query, tf_idf_index = TF_IDFIndex(1)):  #valueType of 1 to help with cosine similarity computation

# Cosine similarity search function

  scores = [(0,0)] * len(os.listdir(path))

  queryVector = QueryVector(query, tf_idf_index)

  for term, idf in queryVector.items():

    for (docID, tf_idf) in tf_idf_index[term]:

      if (docID - 1) in scores:

        scores[docID - 1][1] += idf*tf_idf

        continue

      scores[docID - 1] = (docID, (idf*tf_idf))

  scores = sorted(list(set([score for score in scores])))

  for score in scores:

    docLength = DocLength(queryVector, score[0], tf_idf_index)

    if docLength == 0:

      continue

    score[1] = score[1]/docLength 

  return scores


def main():
    with open('invertedindex.pickle', 'wb') as index:
      pickle.dump(TF_IDFIndex(0), index, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
    
      
  


      






