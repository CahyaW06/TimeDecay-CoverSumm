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
  
  sentences = nltk.sent_tokenize(data[product_id]['review_body'])

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
    representations.append([sentence, data[product_id]['created_at'], emb])
    
  return representations

def get_summarizer(name, method, summary_length=5):
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
                      default='../../../data/raw_reviews/2025/5.json',
                      type=str,
                      help="Path to dataset.")

  args = parser.parse_args()
  
  data_path = Path(args.data_path)
  year = data_path.parent.name
  months = data_path.stem

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModel.from_pretrained(args.model_name)
  model.to(device)

  data = load_json(args.data_path)

  # decay method
  decay_method = ['power', 'exp', 'linear', 'without-decay']

  # report
  report = {} 

  # text segmentation
  text_id = 0
  texts = {}
  for product_id in data.keys():
    representations = load_representations(data, product_id)
    
    for sentence, timestamp, representation in representations:
      input_point = representation.astype(np.float32)
      
      texts[text_id] = {
        'text': sentence,
        'timestamp': timestamp,
        'representation': input_point.tolist()
      }

      text_id += 1
  
  # dump text segmentation
  texts_output_path = f'../../../outputs/text_segmentation/{year}/{months}.json'
  dump_data(texts, texts_output_path)

  # summarization
  for iteration, method in enumerate(decay_method):
    total_time = 0
    review_counter = 0

    if method == 'without-decay': args.summarizer = 'coversumm' 
    else: args.summarizer = 'td_coversumm'
      
    summarizer = get_summarizer(args.summarizer, method=method)
    
    summary_iteration = 1
    summaries = {}

    print(f'========= start summarization with {args.summarizer} using {method} decay method =========')
    for text, timestamp, representation in texts.values():
      review_counter += 1
      start = time()

      input_point = np.array(representation, dtype=np.float32)

      # update summary
      if (args.summarizer == 'td_coversumm'):
        last_summary = summarizer.update_summary(input_point, timestamp)
      else:
        last_summary = summarizer.update_summary(input_point)
      
      # save summary per iteration
      summaries[summary_iteration] = summarizer.get_summary()
      summary_iteration += 1
      
      runtime = time() - start
      total_time += runtime
    
    # get summary text
    full_text_summary = ''
    for idx in summarizer.get_summary():
      full_text_summary += texts[idx]['text'] + ' '
    full_text_summary.strip()

    # dump summary
    summaries_output_path = f'../../../outputs/summaries/{args.summarizer}/{method}/{year}/{months}.json'
    dump_data(summaries, summaries_output_path)

    # save report
    report[iteration] = {
      'summarizer': args.summarizer,
      'dataset': args.data_path,
      'decay_method': summarizer._decay_type if hasattr(summarizer, "_decay_type") else method,
      'decay_rate': summarizer._decay_rate if hasattr(summarizer, "_decay_rate") else None,
      'amortized_runtime': total_time / review_counter,
      'summary_id': list(summarizer.get_summary()),
      'summary_text': full_text_summary,
    }

    print(f'========= end =========')
  
  # dump report
  report_path = f'../../../outputs/reports/{year}/{months}.json'
  dump_data(report, report_path)
  