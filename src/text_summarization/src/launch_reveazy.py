import time
import os
import json
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

import torch
import argparse

import numpy as np

import sys
sys.path.append("../../")

from utils.data import *
from transformers import AutoModel, AutoTokenizer
# from utils.summary import truncate_summary, RougeEvaluator

from tqdm import tqdm
from algorithms.coversumm_summarizer import CoverSummOnlineSummarizer

def load_json(filename):
  with open(filename) as file:
    return json.load(file)

def load_representations(data, product_id):
  representations = []
  splitted_sentences = []
  
  for review in tqdm(data[product_id]):
    sentences = nltk.sent_tokenize(review['review_body'])
    if not sentences:
      continue
    
    batch = tokenizer(sentences,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=True, 
                    max_length=512)
    input_ids = torch.LongTensor(batch['input_ids']).to(device)
    attention_mask = torch.LongTensor(batch['attention_mask']).to(device)
    output = model(input_ids, attention_mask=attention_mask)
    output = output['pooler_output'].detach().cpu().numpy()
    
    representations.append(output)
    for sentence in sentences:
      splitted_sentences.append([sentence, review['created_at']])
  return representations, splitted_sentences

def get_summarizer(name, dim=100):
  summarizer = None
  if name == 'coversumm':
    summarizer = CoverSummOnlineSummarizer()
  return summarizer

def online_summary(points, summarizer=CoverSummOnlineSummarizer(dim=100)):
  for point in points:
    summ = summarizer.update_summary(point)
  return summ

def run_online_summarization(points, summarizer=CoverSummOnlineSummarizer(dim=100)):
  import time # adhoc fix. TODO:find the root cause of this bug
  start = time.time()
  for i in (range(points.shape[0])):
    summ = summarizer.update_summary(points[i])
  return time.time() - start

def np_encoder(object):
  if isinstance(object, np.generic):
    return object.item()

def dump_data(data, path='../../../data/reveazy/output/reveazy_summaries.json'):
  directory = os.path.dirname(path)
  if not os.path.exists(directory):
    os.makedirs(directory)
    
  with open(path, 'w') as f:
    json.dump(data, f, default=np_encoder, indent=4)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--summarizer",
                      default="coversumm",
                      type=str,
                      help="Name of Summarizer.")
  parser.add_argument("--model_name",
                      default="fathan/indojave-codemixed-indobert-base",
                      type=str,
                      help="BERT model name.")
  parser.add_argument("--data_path",
                      default='../../../data/reveazy/reveazy_reviews.json',
                      type=str,
                      help="Path to dataset.")

  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModel.from_pretrained(args.model_name)
  model.to(device)

  data = load_json(args.data_path)

  total_time = 0
  count = 0

  summarizer = get_summarizer(args.summarizer)
  
  text_id = 0
  t_summary = 1
  texts = {}
  summaries = {}

  for product_id in list(data.keys()):
    count += 1
    representations, items = load_representations(data, product_id)
    
    for sentence, timestamp in items:
      texts[text_id] = {
        'text': sentence,
        'timestamp': timestamp
      }
      text_id += 1

    points = representations[0].astype(np.float32)
    
    start = time()
    for point in points:
      last_summary = summarizer.update_summary(point)
    
    full_text_summary = ''
    for idx in summarizer.get_summary():
      full_text_summary += texts[idx]['text'] + ' '
    
    summaries[t_summary] = full_text_summary
    t_summary += 1
    
    runtime = time() - start
    total_time += runtime
  
  texts_output_path = '../../../data/reveazy/output/texts.json'
  dump_data(texts, texts_output_path)

  summaries_output_path = '../../../data/reveazy/output/summaries.json'
  dump_data(summaries, summaries_output_path)

  print(f"Amortized runtime: {total_time / count}")
  