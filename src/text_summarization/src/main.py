import time, json, nltk, torch, argparse, numpy as np, sys, pandas as pd

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

sys.path.append("../../")

from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm
from pathlib import Path
from algorithms.coversumm_summarizer import CoverSummOnlineSummarizer
from algorithms.time_decay_coversumm_summarizer import TimeDecayCoverSummOnlineSummarizer
from functions.dump_json import dump_data
from functions.general import load_json

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def load_representations(data, product_id):
  representations = []
  
  sentences = nltk.sent_tokenize(data[product_id]['review_body'])

  for sentence in sentences:
    # tokenize per kalimat
    inputs = tokenizer(sentence,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        max_length=512,
                        return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # forward pass
    with torch.no_grad():
        output = model(**inputs)
        sent_vec = mean_pool(output.last_hidden_state, inputs['attention_mask'])
        emb = sent_vec.squeeze().cpu().numpy()
    
    # simpan: [kalimat, timestamp, embedding]
    representations.append([sentence, data[product_id]['created_at'], emb])
    
  return representations, len(sentences)

def get_summarizer(name, method, summary_length=10, decay_rate=None):
  summarizer = None
  if method == 'none':
    summarizer = CoverSummOnlineSummarizer(dim=768, summary_length=summary_length)
  else: 
    summarizer = TimeDecayCoverSummOnlineSummarizer(dim=768, summary_length=summary_length, decay_type=method, decay_rate=decay_rate)
  return summarizer

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--summarizer",
                      default="coversumm",
                      type=str,
                      help="Name of Summarizer.")
  parser.add_argument("--model_name",
                      default="fathan/indojave-codemixed-bert-base",
                      type=str,
                      help="BERT model name.")
  parser.add_argument("--data_path",
                      default='../../../data/raw_reviews/2025/6,7,8.json',
                      type=str,
                      help="Path to dataset.")

  args = parser.parse_args()

  print(f'start summarization for {args.data_path}')
  
  data_path = Path(args.data_path)
  year = data_path.parent.name
  months = data_path.stem

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModel.from_pretrained(args.model_name, add_pooling_layer=False).to(device)
  model.eval()

  data = load_json(args.data_path)

  # decay method
  decay_method = [
    ['exp', 0.1],
    ['exp', 0.2],
    ['exp', 0.3],
    ['exp', 0.55],
    ['exp', 0.5],
    ['exp', 0.65],
    ['exp', 0.7],
    ['exp', 0.8],
    ['exp', 0.9],
    ['none', None]
  ]

  # report
  report = {} 

  # text segmentation
  if (Path(f'../../../outputs/text_segmentation/{year}/{months}.json').exists()):
    texts = load_json(f'../../../outputs/text_segmentation/{year}/{months}.json')
  else:
    text_id = 0
    texts = {}
    total_sentences = 0
    for product_id in tqdm(data.keys()):
      representations, sentences_number = load_representations(data, product_id)
      
      total_sentences += sentences_number
      for sentence, timestamp, representation in representations:
        texts[text_id] = {
          'text': sentence,
          'timestamp': timestamp,
          'representation': representation.astype(np.float32).tolist()
        }

        text_id += 1

    # dump text segmentation
    texts_output_path = f'../../../outputs/text_segmentation/{year}/{months}.json'
    dump_data(texts, texts_output_path)

  # sentence_number_for_summary = round(total_sentences / len(data.keys()))
  sentence_number_for_summary = 10

  # summarization
  for iteration, method in enumerate(decay_method):
    total_time = 0
    review_counter = 0

    if method[0] == 'none': args.summarizer = 'coversumm' 
    else: args.summarizer = f'td_coversumm - {method[1]} decay rate'
      
    summarizer = get_summarizer(name=args.summarizer, method=method[0], summary_length=sentence_number_for_summary, decay_rate=method[1] if method[0] != 'none' else None)
    print(summarizer.__class__.__name__)
    
    summary_iteration = 1
    summaries = {}

    try:
      print(f'========= {method} {args.summarizer} =========')
      for text_id, text in texts.items():
        review_counter += 1
        start = time.time()

        input_point = np.array(text['representation'], dtype=np.float32)

        # update summary
        if (args.summarizer == 'coversumm'):
          last_summary = summarizer.update_summary(input_point)
        else:
          last_summary = summarizer.update_summary(input_point, text['timestamp'], text_id)
        
        # save summary per iteration
        summaries[summary_iteration] = summarizer.get_summary()
        summary_iteration += 1
        
        runtime = time.time() - start
        total_time += runtime
    except Exception as e:
      print(f"Error: {e}")
    
    # get summary text
    full_text_summary = ''
    for idx in summarizer.get_summary():
      full_text_summary += texts[f'{idx}']['text'].strip()

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
      'sentence_number_for_summary': sentence_number_for_summary,
      'summary_id': list(summarizer.get_summary()),
      'summary_text': full_text_summary,
    }

    del summarizer
  
  # dump report
  report_path = f'../../../outputs/reports/{year}/{months}.json'
  dump_data(report, report_path)
  