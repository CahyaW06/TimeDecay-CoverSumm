import time, json, nltk, torch, argparse, numpy as np, sys

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

sys.path.append("../../")

from utils.data import *
from transformers import AutoModel, AutoTokenizer
# from utils.summary import truncate_summary, RougeEvaluator

from tqdm import tqdm
from pathlib import Path
from algorithms.coversumm_summarizer import CoverSummOnlineSummarizer
from algorithms.time_decay_coversumm_summarizer import TimeDecayCoverSummOnlineSummarizer
from functions.dump_json import dump_data

def load_json(filename):
  with open(filename) as file:
    return json.load(file)

def load_representations(data, product_id):
  representations = []
  
  for review in tqdm(data[product_id]):
    sentences = nltk.sent_tokenize(review['review_body'])
    if not sentences:
      continue

    batch = tokenizer(sentences,
                      padding='max_length',
                      truncation=True,
                      add_special_tokens=True, 
                      max_length=512,
                      return_tensors='pt')  # lebih rapi
    batch = {k: v.to(device) for k,v in batch.items()}
    output = model(**batch)
    sentence_embeddings = output['pooler_output'].detach().cpu().numpy()
    
    # representations.append(output)
    for sentence, emb in zip(sentences, sentence_embeddings):
      representations.append([sentence, review['created_at'], emb])
  return representations

def get_summarizer(name, method, summary_length=3):
  summarizer = None
  if method == 'without-decay':
    summarizer = CoverSummOnlineSummarizer(summary_length=summary_length)
  else: 
    summarizer = TimeDecayCoverSummOnlineSummarizer(summary_length=summary_length, decay_type=method)
  return summarizer

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
                      default='../../../data/raw_reviews/5.json',
                      type=str,
                      help="Path to dataset.")

  args = parser.parse_args()
  data_path = Path(args.data_path)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModel.from_pretrained(args.model_name)
  model.to(device)

  data = load_json(args.data_path)

  # decay method
  decay_method = ['power', 'exp', 'linear', 'without-decay']

  # report
  report = {} 

  for iteration, method in enumerate(decay_method):
    total_time = 0
    review_counter = 0

    if method == 'without-decay': args.summarizer = 'coversumm' 
    else: args.summarizer = 'td_coversumm'
      
    summarizer = get_summarizer(args.summarizer, method=method)
    
    text_id = 0
    summary_iteration = 1
    texts = {}
    summaries = {}

    for product_id in list(data.keys()):
      review_counter += 1
      start = time()

      representations = load_representations(data, product_id)
      
      for sentence, timestamp, representation in representations:
        input_point = representation.astype(np.float32)
        
        texts[text_id] = {
          'text': sentence,
          'timestamp': timestamp,
        }

        text_id += 1

        # update summary
        if (args.summarizer == 'td_coversumm'):
          last_summary = summarizer.update_summary(input_point, timestamp)
        else:
          last_summary = summarizer.update_summary(input_point)
      
      # get summary text
      full_text_summary = ''
      for idx in summarizer.get_summary():
        full_text_summary += texts[idx]['text'] + ' '
      full_text_summary.strip()
      
      # save summary per iteration
      summaries[summary_iteration] = full_text_summary
      summary_iteration += 1
      
      runtime = time() - start
      total_time += runtime
    
    # dump text segmentation
    texts_output_path = f'../../../outputs/text_segmentation/{data_path.name}.json'
    dump_data(texts, texts_output_path)

    # dump summary
    summaries_output_path = f'../../../outputs/{args.summarizer}/{method}/{data_path.name}.json'
    dump_data(summaries, summaries_output_path)

    print(f"Amortized runtime: {total_time / review_counter}")
    print(f"Text Summary ID: {summarizer.get_summary()}")

    # save report
    report[iteration] = {
      'summarizer': args.summarizer,
      'dataset': data_path,
      'decay_method': summarizer._decay_type if hasattr(summarizer, "_decay_type") else method,
      'decay_rate': summarizer._decay_rate if hasattr(summarizer, "_decay_rate") else None,
      'amortized_runtime': total_time / review_counter,
      'summary_id': list(summarizer.get_summary()),
      'summary_text': full_text_summary,
    }
  
  # dump report
  report_path = f'../../../outputs/reports/report_{data_path.name}.json'
  dump_data(report, report_path)
  