import fitz 
import cv2
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import os

import pytesseract
from pytesseract import Output
import json

import re

from fastapi import FastAPI, UploadFile, File
import uvicorn

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random

#module--1
def preprocess_pdf(pdf_path, output_folder):
    images = convert_from_path(pdf_path, dpi=300, poppler_path=r"C:\poppler\Library\bin")
    os.makedirs(output_folder, exist_ok=True)
    all_img_files = []
    for idx, img in enumerate(images):
        img_g = img.convert('L')
        img_np = np.array(img_g)
        img_deskew = cv2.bitwise_not(img_np)
        img_thresh = cv2.threshold(img_deskew, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img_clean = Image.fromarray(cv2.bitwise_not(img_thresh))
        out_file = os.path.join(output_folder, f'page_{idx+1:02d}.png')
        img_clean.save(out_file)
        all_img_files.append(out_file)
    return all_img_files

#module--2
def run_ocr_and_tokenize(image_path, output_json):
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    tokens = [
        {
            'word': data['text'][i], 
            'left': data['left'][i], 
            'top': data['top'][i], 
            'width': data['width'][i], 
            'height': data['height'][i],
            'conf': data['conf'][i]
        }
        for i in range(len(data['text'])) if data['text'][i].strip()
    ]
    with open(output_json, "w") as f:
        json.dump(tokens, f)
    return tokens

#module--3
def extract_fields(tokens, outfile):
    lines = {}
    for tok in tokens:
        y = tok['top']//10
        if y not in lines: lines[y]=[]
        lines[y].append(tok)
    extracted = {'patient_details': {}, 'tests': []}
    for y, tks in lines.items():
        line = ' '.join([tk['word'] for tk in tks])
        m_name = re.search(r'(Mr\.|Mrs\.|Ms\.|Name)\s*([\w ]+)', line)
        if m_name: extracted['patient_details']['name'] = m_name.group(2).strip()
        m_age = re.search(r'Age\s*([0-9]+)', line)
        if m_age: extracted['patient_details']['age'] = int(m_age.group(1))
        m_gender = re.search(r'(Male|Female)', line)
        if m_gender: extracted['patient_details']['gender'] = m_gender.group(1)
        m_test = re.findall(r'([A-Za-z \-]+)\s([0-9.\-]+)\s([a-zA-Z%/]+)', line)
        for test in m_test:
            extracted['tests'].append({
                'name': test[0].strip(),
                'value': test[1],
                'unit': test[2]
            })
    with open(outfile, 'w') as f:
        json.dump(extracted, f)
    return extracted

#module--4
def confirm_or_correct(extracted_json):
    print("Extracted fields JSON preview:")
    print(json.dumps(extracted_json, indent=2))
    confirmed = input("Confirm (Y) / Edit (E) / Reject (R): ").strip().upper()
    # In a real application, 'E' would trigger an editing interface.
    # For this simulation, we'll just handle 'Y' and 'R'.
    if confirmed == 'Y':
        save_confirmed_json(extracted_json)
    elif confirmed == 'R':
        print("Report rejected.")
    else:
        print("Invalid input. Skipping.")


#module--5

def convert_to_spacy_format(tokens, labels):
    """Converts token/label data to spaCy's entity format."""
    text = " ".join([tok['word'] for tok in tokens])
    entities = []
    current_pos = 0
    for token, label in zip(tokens, labels):
        word = token['word']
        start = text.find(word, current_pos)
        if start == -1: # Should not happen if text is joined correctly
            continue
        end = start + len(word)
        current_pos = end

        if label != 'O': # 'O' means Outside, not an entity
            # The label is expected in BIO format, e.g., B-NAME, I-VALUE
            bio_tag, entity_type = label.split('-')
            if bio_tag == 'B':
                # Start a new entity
                entities.append((start, end, entity_type))
            elif bio_tag == 'I' and entities and entities[-1][2] == entity_type:
                # If it's an 'Inside' tag, extend the last entity
                last_entity = entities[-1]
                entities[-1] = (last_entity[0], end, entity_type)
    
    return (text, {"entities": entities})

def train_token_classifier(train_data, model_path, n_iter=10):
    """
    Trains a spaCy NER model for token classification and saves it.

    Args:
        train_data (list): A list of tuples, where each tuple contains
                           (list_of_token_dicts, list_of_BIO_labels).
        model_path (str): The directory path to save the trained model.
        n_iter (int): Number of training iterations.

    Returns:
        A trained spaCy model object.
    """
    # Start with a blank English model or a pre-trained one
    nlp = spacy.blank("en")
    print("Created blank 'en' model")

    # Create a new NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels from the training data
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    # Start the training
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                try:
                    # Create an Example object
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
                except Exception as e:
                    print(f"Skipping problematic data: {text[:50]}... Error: {e}")
            print(f"Iteration {itn+1}/{n_iter}, Losses: {losses}")

    # Save the trained model to the specified path
    os.makedirs(model_path, exist_ok=True)
    nlp.to_disk(model_path)
    print(f"Saved trained model to {model_path}")

    # --- Validation Step (Optional but Recommended) ---
    # You would typically have a separate `dev_data` set for this.
    # For simplicity, we'll just show how to score on the training data.
    print("\n--- Evaluating on training data ---")
    examples = []
    for text, एनोटेशन्स in train_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, एनोटेशन्स))
    
    scores = nlp.evaluate(examples)
    print("Token-level F1 score:", scores.get('ents_f', 0.0))
    print("Precision:", scores.get('ents_p', 0.0))
    print("Recall:", scores.get('ents_r', 0.0))

    return nlp

#module--6
def run_full_inference(image_path, model_path, rule_extractor):
    tokens = run_ocr_and_tokenize(image_path, "tokens_temp.json")
    result = rule_extractor(tokens, "final_report.json")
    return result

#module--7
app = FastAPI()

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    file_bytes = await file.read()
    with open("temp_report.pdf", "wb") as f:
        f.write(file_bytes)
    images = preprocess_pdf("temp_report.pdf", "./output_images")
    results = []
    for img_file in images:
        tokens = run_ocr_and_tokenize(img_file, f'{img_file}.json')
        result = extract_fields(tokens, f'{img_file}_extracted.json')
        results.append(result)
    return {"results": results}


#module--8
def save_confirmed_json(report_json, folder='final_reports'):
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, f"confirmed_report_{np.random.randint(10000)}.json")
    with open(fname, "w") as f:
        json.dump(report_json, f)
    print("Saved report to", fname)

#module--9
def evaluate(predictions, ground_truth):
    y_true = []
    y_pred = []
    for pred, true in zip(predictions, ground_truth):
        # Simple example: use test names and values
        y_true += [t['name'] for t in true['tests']]
        y_pred += [t['name'] for t in pred['tests']]

    y_true_set = set(y_true)
    y_pred_set = set(y_pred)
    intersection = y_true_set & y_pred_set
    precision = len(intersection) / len(y_pred_set) if y_pred_set else 0
    recall = len(intersection) / len(y_true_set) if y_true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print("Precision:", precision, "Recall:", recall, "F1:", f1)
    return precision, recall, f1

from fastapi import HTTPException

@app.post("/upload")
async def uploadfile(file: UploadFile = File(...)):
    try:
        filebytes = await file.read()
        with open("temp_report.pdf", "wb") as f:
            f.write(filebytes)
        images = preprocess_pdf("temp_report.pdf", "./outputimages")
        results = []
        for imgfile in images:
            tokens = run_ocr_and_tokenize(imgfile, f"{imgfile}.json")
            result = extract_fields(tokens, f"{imgfile}extracted.json")
            results.append(result)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#module--10
# Lab Report Digitisation Assignment

## How to Run
# - Install dependencies: `pip install -r requirements.txt`
# - Run API server: `uvicorn lab_06:app --reload`
# - Upload PDF/JPG/PNG reports, receive JSON results
# - Confirm/correct results for model retraining
