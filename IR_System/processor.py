import imp
from importlib import invalidate_caches
import math
import os
import re
from bs4 import BeautifulSoup
from flask import Flask
from flask import render_template, request, redirect
import pickle
import json
import nltk


html_file_path = r'C:\Users\Chris Hunter\Desktop\Info_Retrieval_Final_Project\html_files'

app = Flask(__name__)

@app.route("/")
def interface():
    return render_template('interface.html')

@app.route('/index', methods=['GET', 'POST'])
def getTF_IDF_Index():
  invertedIndex = TF_IDFIndex(0)
  with open('invertedIndex.pk', 'wb') as pk_file:
    pickle.dump(invertedIndex,pk_file)
  pk_file.close()

  de_pickle = open('invertedIndex.pk', 'rb')

  content = pickle.load(de_pickle)

  json_invertedIndex = json.dumps(content)
  return render_template('invertedIndex.html', inverted_index=json_invertedIndex)


@app.route('/search', methods=['GET', 'POST'])
def search():
  form = request.form
  query = str(form['query'])
  invertedIndex = TF_IDFIndex(1)
  errors = QueryErrors(query, invertedIndex)

  if len(errors) == 0:
    
    legend = fileLegend()
    queryVector = QueryVector(query)
    scores = CSSearch(queryVector)
    return render_template('cssearch.html', legend=legend, queryVector=queryVector, scores=scores)

  corrections = []

  for error in errors:

    corrections.append((error, SpellingCorr(error,invertedIndex)))

  return render_template('interface.html', corrections=corrections)


#------------Indexer Functions----------------

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

def fileLegend():

  filenames ={}
  docID = 0

  for file in os.listdir(html_file_path): 
    
    docID+=1

    file = file.replace(" ", "_")

    filenames[docID] = (f'https://en.wikipedia.org/wiki/{file[:-5]}')

  return filenames

def TF_IDFIndex(valueType):  # valueType indicates the value structure of dictionary; if it's 0 use filenames; if it's 1 use docIDs

  invertedIndex = {} # dictionary to be returned
  corpus = {} # dictionary with html filename as the key and the contents of the html value as the value
  docID = 0

  filenames = fileLegend()
  
  for file in os.listdir(html_file_path): # os.listdir returns a list of the all html files on my local disk
    
    docID+=1

    with open(os.path.join(html_file_path, file), encoding="utf8") as html_file:
      
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

  for key in invertedIndex.keys():
        
        docIDLst = invertedIndex[key]
        
        idf_score = InverseDF(key, len(os.listdir(html_file_path)), invertedIndex)
        
        invertedIndex[key] = []
        
        for docID in docIDLst:
          
          tf_score = TermFrequency(key,corpus[file])

          if valueType != 1:
            
            invertedIndex[key].append((filenames[docID],(tf_score*idf_score)))

          else:

            invertedIndex[key].append((docID,(tf_score*idf_score)))

  return invertedIndex 

def QueryVector(query, tf_idf_index = TF_IDFIndex(1)):

# Query vector function

  qvDict = {}           # Query vector dictionary; will be the final return value

  tokenizedQuery = list(query.split(" "))

  for token in tokenizedQuery:

    if token not in tf_idf_index:

      qvDict[token] = 0
      continue

    qvDict[token] = InverseDF(token, len(os.listdir(html_file_path)), tf_idf_index)

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


def CSSearch(queryVector, tf_idf_index = TF_IDFIndex(1)):  #valueType of 1 to help with cosine similarity computation

# Cosine similarity search function

  scores = [(0,0)] * len(os.listdir(html_file_path))

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

def QueryErrors(query, index):
  
  errors = []

  splitQuery = query.split()

  for term in splitQuery:
    
    if term not in index.keys():
      
      errors.append(term)

  return errors  

def SpellingCorr(query,index):        # Spelling correction functions for free text queries              

  bigramDict = {}          

  bigrams = []

  nwcQuery = query        

  query = "$" + query + "$" 

  for x in range (1,len(query)):
      bigramDict[(query[x-1:x+1]).replace("$","")] = []     
      bigrams.append(query[x-1:x+1]) 
  
  for bigram in bigrams:

    if "$" in bigram[0]:                                   
      for key in index.keys():                             
        if bigram[1] == key[0]:
          bigramDict[bigram[1]].append(key)

    elif "$" in bigram[1]:                                  
      for key in index.keys():
        if bigram[0] == key[-1]:
          bigramDict[bigram[0]].append(key)

    else:
      for key in index.keys():                              
        if bigram in key:                 
          bigramDict[bigram].append(key)

  bigrams.clear()
  
  for x in range (1,len(query)):                            
    bigrams.append((query[x-1:x+1]).replace("$",""))

 #-------------------------------------Get edit distances        

  editDists = []                    # 2D array of edit distances

  minDists = []                     # 1D array of the minimum edit distances

  for bigram in bigrams:
    editDist = []                   # 1D array of edit distances
    for term in bigramDict[bigram]:
      editDist.append(nltk.edit_distance(nwcQuery,term))
    editDists.append(editDist)

  for lst in editDists:
    if len(lst) == 0:
      continue
    minDists.append(min(lst))

#------------------------------------Edit distances obtained; generating suggestions

  minDist = min(minDists)

  suggestions = []           # A list of potential spelling suggestions

  pointer = -1                # Tracks list position in editDists
 
  for lst in editDists:
    pointer+=1
    for x in range(0,len(lst)):
      if lst[x] == minDist:
        suggestions.append(bigramDict[bigrams[pointer]][x])

  return list(set(suggestions))        # Removing duplicate suggestions and returning them
    
#-----------------End of Indexer Functions-----------------------------

if __name__ == '__main__':
  app.run(debug=True)
