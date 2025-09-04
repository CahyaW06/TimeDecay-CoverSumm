import pymysql, os, argparse, re
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append("../")

from src.functions.dump_json import dump_data

DB_HOST=os.environ.get('DB_HOST')
DB_NAME=os.environ.get('DB_NAME')
DB_USER=os.environ.get('DB_USER')
DB_PASSWORD=os.environ.get('DB_PASSWORD')
DB_PORT=os.environ.get('DB_PORT')

def clean_duplicate_punc(text: str):
    return re.sub(r'([!?.,:;])\1+', r'\1', text)

def replace_newline_with_dot(text: str):
    return text.replace('\n', '.')

def remove_emoji(text: str):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  
        "\U0001F780-\U0001F7FF"  
        "\U0001F800-\U0001F8FF"  
        "\U0001F900-\U0001F9FF"  
        "\U0001FA00-\U0001FA6F"  
        "\U0001FA70-\U0001FAFF"  
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_non_ascii(text: str):
    return text.encode("ascii", "ignore").decode()
    
def preprocess(text: str):
    text = remove_emoji(text)
    text = remove_non_ascii(text)
    text = clean_duplicate_punc(text)
    text = replace_newline_with_dot(text)
    return text

def get_reviews_from_database(months: list, year: str):
    months_str = ''
    for month in months:
        months_str += f"{month},"
    months_str = months_str.rstrip(',')

    db = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME, port=int(DB_PORT))
    cursor = db.cursor()
    
    cursor.execute(f"SELECT raw_content, created_at FROM reviews WHERE raw_content IS NOT NULL AND MONTH(created_at) IN ({months_str}) AND YEAR(created_at) = {year}")
    
    reviews = cursor.fetchall()
    db.close()
    return reviews, months_str, year

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--months",
                        nargs='+',
                        type=int,
                        help="List of months to be selected.")
    parser.add_argument("--year",
                        type=str,
                        help="Year to be selected.")
    
    args = parser.parse_args()

    reviews, months_str, year = get_reviews_from_database(args.months, args.year)
    
    review_dict = {}
    for idx, item in enumerate(reviews):
        review_body = preprocess(item[0])

        if not review_body.strip(): 
            continue
        
        review_dict[idx] = {
            'review_body': review_body,
            'created_at': item[1].isoformat()
        }

    data_path = f'../data/raw_reviews/{year}/{months_str}.json'
    dump_data(review_dict, data_path)

    print(f"Data saved to {data_path}")
