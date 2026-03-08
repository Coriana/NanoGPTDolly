"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time
import random
import pyttsx3
import concurrent.futures
import sys
from time import sleep
import socket
import re
import sqlite3
from history import History, DonationHistory, FollowHistory

# Connect to the SQLite database
conn = sqlite3.connect('samples.db')
c = conn.cursor()

# Create a table to store the samples
c.execute('''CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    history_data TEXT NOT NULL
);''')

c.execute('''CREATE TABLE IF NOT EXISTS samples (
    history_id INTEGER,
    sample_data TEXT NOT NULL,
    rejected_sample TEXT NOT NULL,
    FOREIGN KEY (history_id) REFERENCES history(id)
);
''')

c.execute('''CREATE VIEW IF NOT EXISTS history_samples AS
SELECT history.id AS history_id, history.history_data, samples.sample_data, samples.rejected_sample
FROM history
JOIN samples ON history.id = samples.history_id;''')
MAX_HISTORY_LENGTH = 768


def send_data(data, port):
    s = socket.socket()
    s.connect(('localhost', port))
    s.send(data.encode())
    s.close()
    
def gentext(history):
    start_ids = encode(add_caseifer(history))
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    generated_data = ''
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        output = remove_caseifer(decode(y[0].tolist()))[len(history):]
        generated_data = output.split("\n", 1)[0]
        if generated_data == '\n':
            generated_data = output.split("\n", 2)[0]
            if generated_data == '\n':
                generated_data = output.split("\n", 3)[0]
       # while not generated_data.strip():  # loop until parsed_output contains non-empty characters
       #     generated_data = generated_data[generated_data.find("\n") + 1:]  # find the first newline character and remove it from the parsed_output

        generated_sequences.append(generated_data)
    
    #for i, sequence in enumerate(generated_sequences):
        #print(f"Output {i}: {sequence}")    
    
    #selected_output = int(input("Select output number to use: "))         
    selected_output = random.randrange(num_samples) 
    actual_output = generated_sequences[selected_output]
    return actual_output

def typing(text):
    for char in text:
        sleep(0.001)
        sys.stdout.write(char)
        sys.stdout.flush()
    sys.stdout.write('\n')

# Create a list of names
names = ["Alice", "Eml"]

# Use the random.choice() function to select a random name from the list
        
def textToSpeech(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) # female voice
    engine.setProperty('rate', 175) # change the speaking rate
    engine.say(text)
    engine.runAndWait()
    del engine

def parallel(text):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_tasks = {executor.submit(textToSpeech, text), executor.submit(typing, text)}
        for future in concurrent.futures.as_completed(future_tasks):
            try:
                data = future.result()
            except Exception as e:
                print(e)
                
def remove_caseifer(text):
    new_text = ""
    i = 0
    while i < len(text):
        if text[i] == "↨":
            if i+1 < len(text):
                new_text += text[i+1].upper()
                i += 1
            else:
                pass  # skip this index
        else:
            new_text += text[i]
        i += 1
    return new_text
    
def add_caseifer(text):
    new_text = ""
    for char in text:
        if char.isupper():
            new_text += "↨" + char.lower()
        else:
            new_text += char
    return new_text
    
#def Read_input():
    # do this

#def ContinueOutput():
    # do that

#def StartFromNothing():
    # do the other thing
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'Finetuned' # ignored if init_from is not 'resume'
start = "Corianas just followed the channel." # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 512 # number of tokens generated in each sample
temperature = 0.9 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = None # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1336
MAX_LENGTH = 1024
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
direction_file = "files\\direction.txt"
new_donation_file = "files\\new_donation.txt"
new_follower_file = "files\\new_follower.txt"
follower_file = "files\\follower.txt"
input_file = "files\\input.txt"
autorun_file = "files\\autoplay.txt"
sample_file = "files\\sampler.txt"

#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
past_history = History()
donate_history = DonationHistory()
follow_history = FollowHistory()

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
# look for the meta pickle in case it is available in the dataset folder

load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join(out_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    #print(meta)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
        

MAX_LASTHISTORY_LENGTH = 3

# Initialize the last 10 strings said to empty strings
last_said = [''] * MAX_LASTHISTORY_LENGTH
dontsay = ['']
dontsay.append('.')
forbidden_words = ''
with open("files\\badwords.txt", "r") as f:
    forbidden_words = [line.strip() for line in f.readlines()]
    
badwords = [re.compile(rf"\b{word}\b", re.IGNORECASE) for word in forbidden_words]

start_ids = encode(add_caseifer(start))
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
history = start
#lastsaid = start
generated_sequences = []
#past_history.name = "I am Emeldar and "
past_history.direction = "You are an AI called Emeldar, interact with your chat/followers and make them feel welcome."
#past_history.add(start)
#past_history.add('Liam: And so– so, you keep saying that this is about brooke, but–\nDouglas: It’s true. I saw them.')
print(str(past_history))
# run generation
GEN = True
shift = False
wasinputted = False
MAX_HISTORY_LENGTH = model.config.block_size - 80 # Memory size minus approx reply. (ish)
with torch.no_grad():
    with ctx:
        while True:
            if os.path.getsize(sample_file) > 0:
                with open(sample_file, "r") as f:
                    num_samples = int(file.readline())
                    
            if os.path.getsize(new_donation_file) > 0:
                with open(new_donation_file, "r") as f:
                    lines = f.readlines()
                inputted_data = lines[0]
               # clean = re.sub(r"[\n\r\t\v\f]", " ", inputted_data)
                lines.pop(0)
                with open(new_donation_file, "w") as f:
                    f.writelines(lines)
                    
                if not any(pattern.search(inputted_data) for pattern in badwords):
                    donate_history.add(inputted_data)
                    print(inputted_data)

                x = (torch.tensor(encode(add_caseifer(str(donate_history))), dtype=torch.long, device=device)[None, ...])
                for k in range(num_samples):
                    for idx_next in model.generate_streaming(x, max_new_tokens, temperature=temperature):
                        # convert the index to a character and print it to the screen
                        char = decode([idx_next])

                        # check for newline character
                        if char == '\n':
                            # append the completed line to the list or print it to the screen
                       #     generated_sequences.append(generated_text)
                            # reset the generated text for the next line
                            generated_data = remove_caseifer(generated_text)
                            generated_text = ''
                            break
                        generated_text += char
                    generated_sequences.append(generated_data)

                if (os.path.isfile(sample_file) and os.path.getsize(sample_file) > 0) and num_samples > 1:
                    for i, sequence in enumerate(generated_sequences):
                        print(f"Output {i}: {sequence}")
                    selected_output = int(input("Select output number to use: "))
                    actual_output = generated_sequences.pop(selected_output)  # Remove the actual output from the list.
                    # Insert the history data into the history table
                    c.execute("INSERT INTO history (history_data) VALUES (?)", (history,))
                    history_id = c.lastrowid  # Get the ID of the inserted history row
                    # Insert the data into the samples table
                    for rejected_sample in generated_sequences:
                        c.execute("INSERT INTO samples (history_id, sample_data, rejected_sample) VALUES (?, ?, ?)", (history_id, actual_output, rejected_sample))  # Set rejected flag to True.
                else:
                    actual_output = generated_sequences[random.randrange(num_samples)]



                donate_history.add(actual_output)    
                try:
                   # send_data(inputted_data, 1234)   
                    send_data(actual_output, 1234)
                except:
                    pass
                print(actual_output)
                continue


            if os.path.getsize(new_follower_file) > 0:
                with open(new_follower_file, "r") as f:
                    lines = f.readlines()
                inputted_data = lines[0]
               # clean = re.sub(r"[\n\r\t\v\f]", " ", inputted_data)
                lines.pop(0)
                with open(new_follower_file, "w") as f:
                    f.writelines(lines)
                    
                if not any(pattern.search(inputted_data) for pattern in badwords):
                    follow_history.add(inputted_data)
                    print(inputted_data)

                x = (torch.tensor(encode(add_caseifer(str(follow_history))), dtype=torch.long, device=device)[None, ...])
                for k in range(num_samples):             
                    for idx_next in model.generate_streaming(x, max_new_tokens, temperature=temperature, top_k=top_k):
                        # convert the index to a character and print it to the screen
                        char = decode([idx_next])

                        # check for newline character
                        if char == '\n':
                            # append the completed line to the list or print it to the screen
                       #     generated_sequences.append(generated_text)
                            # reset the generated text for the next line
                            generated_data = remove_caseifer(generated_text)
                            generated_text = ''
                            break
                        generated_text += char
                    generated_sequences.append(generated_data)

                if (os.path.isfile(sample_file) and os.path.getsize(sample_file) > 0) and num_samples > 1:
                    for i, sequence in enumerate(generated_sequences):
                        print(f"Output {i}: {sequence}")
                    selected_output = int(input("Select output number to use: "))
                    actual_output = generated_sequences.pop(selected_output)  # Remove the actual output from the list.
                    # Insert the history data into the history table
                    c.execute("INSERT INTO history (history_data) VALUES (?)", (history,))
                    history_id = c.lastrowid  # Get the ID of the inserted history row
                    # Insert the data into the samples table
                    for rejected_sample in generated_sequences:
                        c.execute("INSERT INTO samples (history_id, sample_data, rejected_sample) VALUES (?, ?, ?)", (history_id, actual_output, rejected_sample))  # Set rejected flag to True.

                else:
                    actual_output = generated_sequences[random.randrange(num_samples)]



                follow_history.add(actual_output)    
                try:
                   # send_data(inputted_data, 1234)   
                    send_data(actual_output, 1234)
                except:
                    pass
                print(actual_output)
                continue

                
            if os.path.isfile(direction_file) and os.path.getsize(direction_file) > 0:
                with open(direction_file, "r") as f:
                    past_history.direction = f.read()
                with open(direction_file, "w") as f:
                    f.writelines('')
               # GEN = True
                    
            if os.path.isfile(follower_file) and os.path.getsize(follower_file) > 0:
                with open(follower_file, "r") as f:
                     lines = f.readlines()

                # Select a random line from the file and remove it
                #random_line = random.choice(lines)
                #lines.remove(random_line)
                inputted_data = lines[0]
               # clean = re.sub(r"[\n\r\t\v\f]", " ", inputted_data)
                lines.pop(0)
                with open(follower_file, "w") as f:
                    f.writelines(lines)
                if not any(pattern.search(inputted_data) for pattern in badwords):
                    past_history.add(inputted_data)
                    wasinputted = False

                    GEN = True                      
                    
            if os.path.isfile(input_file) and os.path.getsize(input_file) > 0 and GEN == False:
                with open(input_file, "r") as f:
                     lines = f.readlines()

                # Select a random line from the file and remove it
                #random_line = random.choice(lines)
                #lines.remove(random_line)
                inputted_data = lines[0]
               # clean = re.sub(r"[\n\r\t\v\f]", " ", inputted_data)
                lines.pop(0)
                with open(input_file, "w") as f:
                    f.writelines(lines)
                if not any(pattern.search(inputted_data) for pattern in badwords):
                    past_history.add(inputted_data)
                    print(inputted_data)
                    wasinputted = False
                    GEN = True                      
                
            if random.randint(1, 50) == 45:
                GEN = True
            else:
                time.sleep(0.5)
                
                
            if os.path.isfile(autorun_file) and os.path.getsize(autorun_file) > 0:
                GEN = True
                
            if GEN:
                #random_name = random.choice(names)

                history = str(past_history) + 'Emeldar: ' # append input to history
                #print(history,end='', flush=True)
                try:
                    start_ids = encode(add_caseifer(history))
                except:
                    continue
               # print(history)
                x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                generated_data = ''
                generated_text = ''
                for k in range(num_samples):
                    for idx_next in model.generate_streaming(x, max_new_tokens, temperature=temperature, top_k=top_k):
                        # convert the index to a character and print it to the screen
                        char = decode([idx_next])

                        # check for newline character
                        if char == '\n':
                            # append the completed line to the list or print it to the screen
                       #     generated_sequences.append(generated_text)
                            # reset the generated text for the next line
                            generated_data = remove_caseifer(generated_text)
                            generated_text = ''
                            break
                        generated_text += char


                    while any(generated_data == s for s in last_said) or any(generated_data == s for s in dontsay) or any(pattern.search(generated_data) for pattern in badwords):
                        for idx_next in model.generate_streaming(x, max_new_tokens, temperature=temperature, top_k=top_k):
                            # convert the index to a character and print it to the screen
                            char = decode([idx_next])
                            # check for newline character
                            if char == '\n':
                                # append the completed line to the list or print it to the screen
                            #     generated_sequences.append(generated_text)
                                # reset the generated text for the next line
                                generated_data = remove_caseifer(generated_text)
                                generated_text = ''
                                break
                            generated_text += char

                  #  print(f"Output {k}: {generated_data}")    
                    generated_sequences.append(generated_data)

                if (os.path.isfile(sample_file) and os.path.getsize(sample_file) > 0) and num_samples > 1:
                    for i, sequence in enumerate(generated_sequences):
                        print(f"Output {i}: {sequence}")
                    selected_output = int(input("Select output number to use: "))
                    actual_output = generated_sequences.pop(selected_output)  # Remove the actual output from the list.
                    # Insert the history data into the history table
                    c.execute("INSERT INTO history (history_data) VALUES (?)", (history,))
                    history_id = c.lastrowid  # Get the ID of the inserted history row
                    # Insert the data into the samples table
                    for rejected_sample in generated_sequences:
                        c.execute("INSERT INTO samples (history_id, sample_data, rejected_sample) VALUES (?, ?, ?)", (history_id, actual_output, rejected_sample))  # Set rejected flag to True.

                else:
                    actual_output = generated_sequences[random.randrange(num_samples)]

                past_history.add(actual_output)
              #  past_history.add(random_name + ': ' + actual_output+ '\n')
               # print(random_name + ': ' + actual_output)
              #  past_history.add(actual_output)
                #history = history + actual_output + '\n'
                #history = history[:MAX_LENGTH]
                if wasinputted:
                    try:
                        print(inputted_data)
                        print(actual_output)
                        send_data(inputted_data +' ' + actual_output, 1234)
                    except:
                        failed = 1
                 #   typing(inputted_data + actual_output)
                    wasinputted = False
                    inputted_data = ''
                else:
                    try:
                        print(actual_output)
                        send_data(actual_output, 1234)
                    except:
                        failed = 1
                  #  typing(actual_output)
                #print(actual_output)
                last_said.append(actual_output)
                last_said.pop(0)
                #if actual_output.endswith('.'):
                #    sleep_time = random.uniform(0.25, 2)
                #    time.sleep(sleep_time)
                #elif actual_output.endswith('?'):
                #    sleep_time = random.uniform(0.5, 3)
                #    time.sleep(sleep_time)
                #parallel(actual_output)
                generated_sequences = []
                GEN = False
                # Commit the changes and close the connection
            conn.commit()
        conn.close()
