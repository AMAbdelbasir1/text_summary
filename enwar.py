import spacy
from flask import Flask, request, jsonify
from heapq import nlargest
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from langdetect import detect
import re
import os

app = Flask(__name__)

# Load the English language model
nlp_en = spacy.load('en_core_web_sm')

def normalize_frequencies(word_frequencies):
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency
    return word_frequencies

def calculate_sentence_scores(doc, word_frequencies):
    sentence_scores = {}
    for sentence in doc.sents:
        for word in sentence:
            word_lower = word.text.lower()
            if word_lower in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word_lower]
    return sentence_scores

def summarize_text_english(text, summary_ratio=0.3):
    doc = nlp_en(text)
    word_frequencies = {}
    for word in doc:
        word_lower = word.text.lower()
        if word_lower not in nlp_en.Defaults.stop_words and not word.is_punct:
            word_frequencies[word_lower] = word_frequencies.get(word_lower, 0) + 1

    if not word_frequencies:
        return ""

    word_frequencies = normalize_frequencies(word_frequencies)
    sentence_scores = calculate_sentence_scores(doc, word_frequencies)

    num_sentences = max(1, int(len(list(doc.sents)) * summary_ratio))
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    final_summary = ' '.join([sentence.text for sentence in summary_sentences])

    return final_summary

def summarize_text_arabic(text, summary_ratio=0.3):
    text = dediac_ar(text)  # Remove diacritics
    words = simple_word_tokenize(text)
    word_frequencies = {}
    for word in words:
        word_frequencies[word] = word_frequencies.get(word, 0) + 1

    if not word_frequencies:
        return ""

    word_frequencies = normalize_frequencies(word_frequencies)

    sentences = text.split('.')
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = simple_word_tokenize(sentence)
        for word in sentence_words:
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

    num_sentences = max(1, int(len(sentences) * summary_ratio))
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    final_summary = ' '.join(summary_sentences)

    return final_summary

def summarize_text(text, lang='en', summary_ratio=0.3):
    if lang == 'en':
        return summarize_text_english(text, summary_ratio)
    elif lang == 'ar':
        return summarize_text_arabic(text, summary_ratio)
    else:
        return "Language not supported"

def clean_response(text):
    return re.sub(r'[\r\n"]', '', text)

@app.route('/summarize', methods=['POST'])
def handle_summarize_request():
    try:
        uploaded_file = request.files.get('file')
        if not uploaded_file or uploaded_file.filename == '':
            return jsonify({'status': 'fail', 'error': 'No file uploaded or selected'}), 400

        text = uploaded_file.read().decode('utf-8')
        language = detect(text)

        if language not in ['en', 'ar']:
            return jsonify({'status': 'fail', 'error': 'Language not supported'}), 400

        summary = summarize_text(text, lang=language)
        cleaned_summary = clean_response(summary)

        return jsonify({
            'status': 'success',
            'summary': cleaned_summary,
            'lengthSUMMARY': len(cleaned_summary),
            'lengthTEXT': len(text),
            'language': language,
        }), 200
    except Exception as e:
        return jsonify({'status': 'fail', 'error': 'Something went wrong, please try again'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
