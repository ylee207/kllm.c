import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import logging
import sys

from data_common import write_datafile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for different datasets
DATASET_CONFIGS = {
    "HAERAE-HUB/KOREAN-WEBTEXT": {
        "dataset": "HAERAE-HUB/KOREAN-WEBTEXT",
        "config": None,
        "split": "train",
        "text_column": "text",
        "shard_size": 10**8,
        "output_dir": "processed_data"
    },
     "HAERAE-HUB/KOREAN-SyntheticText-1.5B": {
        "dataset": "HAERAE-HUB/KOREAN-SyntheticText-1.5B",
        "config": None,
        "split": "train",
        "text_column": "text",
        "shard_size": 10**8,
        "output_dir": "processed_data"
    },
      "HAERAE-HUB/QARV-KOEN-1B-Entangled": {
        "dataset": "HAERAE-HUB/QARV-KOEN-1B-Entangled",
        "config": None,
        "split": "train",
        "text_column": "text",
        "shard_size": 10**8,
        "output_dir": "processed_data"
    },
      "HAERAE-HUB/QARV-KOEN-1B-Disentangled": {
        "dataset": "HAERAE-HUB/QARV-KOEN-1B-Disentangled",
        "config": None,
        "split": "train",
        "text_column": "text",
        "shard_size": 10**8,
        "output_dir": "processed_data"
    },
    # Add more dataset configurations as needed
}

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token

def tokenize(doc):
    try:
        tokens = [eot]  # the special <|endoftext|> token delimits all documents
        tokens.extend(enc.encode_ordinary(doc["text"]))  # Use a default key "text" here
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        return tokens_np.astype(np.uint16)
    except Exception as e:
        logging.error(f"Error tokenizing document: {str(e)}")
        return np.array([], dtype=np.uint16)

def process_dataset(config_name):
    config = DATASET_CONFIGS[config_name]
    
    logging.info(f"Starting to process dataset: {config_name}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    # Load the dataset
    try:
        logging.info(f"Loading dataset: {config['dataset']}")
        if config["config"]:
            dataset = load_dataset(config["dataset"], config["config"], split=config["split"])
        else:
            dataset = load_dataset(config["dataset"], split=config["split"])
        logging.info(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
    except Exception as e:
        logging.error(f"Error loading dataset {config['dataset']}: {str(e)}")
        return

    # Tokenize all documents and write output shards
    nprocs = max(1, os.cpu_count() - 2)
    logging.info(f"Starting tokenization with {nprocs} processes")
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((config["shard_size"],), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in tqdm(pool.imap(tokenize, dataset, chunksize=16), total=len(dataset), desc="Tokenizing", file=sys.stdout):
            if len(tokens) == 0:
                continue  # Skip empty results from tokenization errors
            if token_count + len(tokens) < config["shard_size"]:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=config["shard_size"], unit="tokens", desc=f"Shard {shard_index}", file=sys.stdout)
                progress_bar.update(len(tokens))
            else:
                remainder = config["shard_size"] - token_count
                if progress_bar:
                    progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                
                filename = os.path.join(config["output_dir"], f"train_{shard_index:06d}.bin")
                
                write_datafile(filename, all_tokens_np)
                logging.info(f"Wrote shard {shard_index} to {filename}")
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # Write any remaining tokens as the last shard
        if token_count != 0:
            filename = os.path.join(config["output_dir"], f"train_{shard_index:06d}.bin")
            
            write_datafile(filename, all_tokens_np[:token_count])
            logging.info(f"Wrote final shard {shard_index} to {filename}")

    logging.info(f"Finished processing {config_name}")

def main():
    if len(sys.argv) > 1:
        dataset_names = sys.argv[1:]
        for dataset_name in dataset_names:
            if dataset_name in DATASET_CONFIGS:
                process_dataset(dataset_name)
            else:
                logging.error(f"Unknown dataset: {dataset_name}")
    else:
        logging.info("No dataset specified. Processing all available datasets.")
        for dataset_name in DATASET_CONFIGS:
            process_dataset(dataset_name)

if __name__ == "__main__":
    main()