# Computational Linguistics & Natural Language Processing (NLP) Lab

Welcome to the **Computational Linguistics and Natural Language Processing (NLP) Laboratory** repository. This repository is dedicated to exploring the core foundations of data analysis, exploratory data visualization, and fundamental text preprocessing pipelines in NLP. 

The codebase is split into three primary laboratories:
1. **Lab 1**: Exploratory Data Analysis (EDA) and Visualization on Tabular Datasets.
2. **LAB 2**: NLP Text Preprocessing, Word/Sentence Tokenization (using NLTK, spaCy, and Regex), and Stop Words Filtering.
3. **Lab 3**: Stemming, Lemmatization, and Regular Expression Pattern Matching.
4. **Lab 4**: Advanced Text Processing & Document Modeling
5. **LAB 5**: Advanced Subword Tokenization, POS Tagging, & Word Vector Embeddings
---

---

## 🔬 Lab Details

### 📊 Lab 1: Exploratory Data Analysis & Visualization
This lab focuses on data cleaning, grouping, statistical analysis, and visual storytelling using **Pandas** and **Matplotlib**. It serves as a foundational step for understanding data distributions before feeding them into computational pipelines.

#### Experiments Covered:
*   **Experiment 1.1: Average Salary by Department**
    *   *Concept*: Data grouping and aggregation.
    *   *Implementation*: Compute average salaries per department and display them using a matplotlib bar chart.
*   **Experiment 1.2: Department Headcount**
    *   *Concept*: Categorical distributions.
    *   *Implementation*: Display employee headcount across departments using value counts.
*   **Experiment 1.3: Gender Diversity Analysis**
    *   *Concept*: Proportion representation.
    *   *Implementation*: Render a pie chart displaying the percentage breakdown of male vs. female employees.
*   **Experiment 1.4: Salary Distribution Histogram**
    *   *Concept*: Continuous variables distributions.
    *   *Implementation*: Draw a histogram to identify skewness, range, and common salary bands.
*   **Experiment 1.5: Experience vs. Salary Scatter Plot**
    *   *Concept*: Correlation and regression insights.
    *   *Implementation*: Map professional experience (years) against salary to visualize compensation growth patterns.
*   **Experiment 1.6: Top Earners Identification**
    *   *Concept*: Sorting and filtering.
    *   *Implementation*: Sort dataset by salary descending to extract the top 10 highest-paid employees.
*   **Experiment 1.7: Departmental Peak Compensation**
    *   *Concept*: Grouped max-aggregation.
    *   *Implementation*: Identify the maximum salary ceiling within each department.
*   **Experiment 1.8: Above-Average Earners**
    *   *Concept*: Threshold filtering.
    *   *Implementation*: Compute global average salary and filter list of employees earning above it.
*   **Experiment 1.9: Average Tenure by Department**
    *   *Concept*: Departmental experience profiles.
    *   *Implementation*: Compute the mean years of experience grouped by department.
*   **Experiment 1.10: Age Distribution Histogram**
    *   *Concept*: Demographic distribution.
    *   *Implementation*: Plot age frequency distribution using a histogram.

---

### 🔤 LAB 2: Computational Linguistics & NLP Preprocessing
This lab introduces the fundamental processing layers required for any Natural Language Processing pipeline. Text data is inherently unstructured; preprocessing cleanses and breaks text down into tokens suitable for language modeling.

#### 1. Experiment 2.1: Basic Text Preprocessing Pipeline
Text cleansing is essential to reduce vocabulary size and normalize characters. This experiment builds a pipeline that performs:
1.  **Lowercasing**: Normalizes casing (e.g., "Apple" and "apple" become identical) while counting uppercase letters.
2.  **Punctuation Removal**: Strips structural symbols (e.g., `!`, `,`, `.`) and outputs the count of unique punctuation characters removed.
3.  **Number Removal**: Strips numerical characters (`0-9`) to isolate linguistic components.
4.  **Whitespace Normalization**: Collapses multiple spaces, tabs, or newlines down to a single space.
5.  **Output Visualisation**: Displays side-by-side comparisons of raw and cleaned texts.

#### 2. Experiment 2.2: Tokenization (Word & Sentence Level)
Tokenization is the task of breaking down a block of text into individual linguistic units (tokens) such as words, phrases, or sentences. This experiment contrasts three different technical implementations:
*   **NLTK (Natural Language Toolkit)**:
    *   Uses rule-based algorithms like `PunktSentenceTokenizer` for sentence splitting and Penn Treebank-style rules for word tokenization.
*   **spaCy**:
    *   An industrial-strength library that utilizes a state-of-the-art dependency parsing pipeline. It models tokenization using language-specific rules and non-destructive tokenizer graphs.
*   **Custom Regular Expressions (Python `re`)**:
    *   A zero-dependency approach using regex patterns (e.g., splitting on punctuation followed by spaces `(?<=[.!?])\s+`) to demonstrate the inner workings of tokenizers.

#### 3. Experiment 2.3: Tokenization & Stop Words Removal
Stop words (e.g., "is", "the", "a", "an", "to") carry high frequency but low semantic information. Removing them helps algorithms focus on content words (nouns, verbs, adjectives).
*   **Implementation**:
    *   Reads `2.3_clean_data.txt`.
    *   Tokenizes the input string.
    *   Filters tokens against the NLTK stop words dictionary.
    *   Outputs the original tokens, filtered tokens, and isolates the stop words that were stripped out.


### 🔬 Lab 3: Stemming, Lemmatization, and Regular Expression Pattern Matching

This laboratory focuses on morphological normalization techniques (Stemming and Lemmatization) and pattern-based information extraction using Regular Expressions. These techniques are crucial for reducing vocabulary dimensionality and extracting structured data from unstructured text.

#### 1. Experiment 3.1: Stemming using Porter Stemmer
Stemming is a heuristic process that chops off the ends of words to reduce them to their common base form or "stem". This experiment utilizes the classic **Porter Stemmer** via the NLTK library to demonstrate rule-based suffix stripping.
*   **Concept**: Rule-based inflectional reduction.
*   **Implementation**:
    *   Loads a target list of inflected words (including verbs, plurals, and gerunds like *playing*, *studies*, *computers*).
    *   Applies the Porter Stemming algorithm sequentially.
    *   Outputs a side-by-side comparative table demonstrating how grammatical inflections are mapped back to their base stems (e.g., *studies* $\rightarrow$ *studi*, *connection* $\rightarrow$ *connect*).
    *   *Observation*: Highlights common issues like over-stemming and under-stemming where the resulting stem is not a valid dictionary word.

#### 2. Experiment 3.2: Lemmatization with POS Tagging
Unlike stemming, lemmatization uses vocabulary and morphological analysis to return the dictionary form of a word, known as the *lemma*. This experiment implements NLTK's **WordNetLemmatizer** to contrast how lemma extraction behaves across different Parts of Speech (POS).
*   **Concept**: Vocabulary-accurate morphological normalization.
*   **Implementation**:
    *   Iterates through a diverse set of words containing irregular plurals and verb tenses (e.g., *mice*, *went*, *better*, *leaves*).
    *   Applies lemmatization across three distinct POS parameters: **Noun (`pos='n'`)**, **Verb (`pos='v'`)**, and **Adjective (`pos='a'`)**.
    *   Outputs a comprehensive matrix displaying the varying structural outputs.
    *   *Observation*: Illustrates how lemmatization preserves actual linguistic words and demonstrates the high dependency of lemmas on correct grammatical context (e.g., *better* as an adjective maps to *good*, *went* as a verb maps to *go*).

#### 3. Experiment 3.3: Regular Expressions (Regex) for Information Extraction
Regular expressions provide an extremely powerful, zero-dependency mechanism for parsing, validating, and extracting specific structural patterns from raw unstructured text data.
*   **Concept**: Pattern matching and information extraction.
*   **Implementation**:
    *   Parses a raw multi-line text corpus containing contact details, social links, and metadata.
    *   Implements customized regex engines using Python's `re` module to capture:
        1.  **Email Addresses**: Using character-class boundaries matching structural `@` domains.
        2.  **URLs**: Matching protocols (`http`, `https`) and `www` subdomains.
        3.  **Mobile Numbers**: Identifying international prefixes (e.g., `+91-`) and standard 10-digit formats.
        4.  **Hashtags**: Isolating metadata tags prefixed with `#`.
        5.  **Mentions**: Tracking user handles starting with `@` symbols.
    *   Outputs isolated, clean arrays of structured metadata extracted directly from the unstructured text blocks.

### 🔍 Lab 4: Advanced Text Processing & Document Modeling
This lab covers the essential methods used in search engines, text mining, and information retrieval. It covers word-frequency distributions, Named Entity Recognition (NER) utilizing modern NLP pipelines, and manual tf-idf construction for information extraction.

#### 1. Experiment 4.1: Term-Frequency Analysis & Named Entity Recognition (with toolkits)
This experiment utilizes industrial-strength NLP libraries to tokenize texts, compute term frequencies, and identify semantic entities.
*   **Term-Frequency (TF) Analysis (NLTK)**:
    *   *Implementation*: Tokenizes the raw input text using NLTK's `word_tokenize`, filters non-alphabetic tokens, and aggregates frequencies using a `Counter` dictionary to output a structured `.csv` alongside the top 10 most frequent terms.
*   **Named Entity Recognition (spaCy)**:
    *   *Implementation*: Employs the `en_core_web_sm` model to parse unstructured corpora and extract semantic tags, cataloging entities and their corresponding types (e.g., PERSON, ORG, GPE, DATE).

#### 2. Experiment 4.2: Term-Frequency Analysis (without any toolkit)
To fully grasp tokenization mechanics, this experiment builds a basic dictionary loader and string parser without the aid of existing natural language packages.
*   **Implementation**:
    *   Loads raw text files, normalizes all characters to lowercase, and strips punctuation using native Python `string.punctuation` mapping structures.
    *   Manually tokenizes strings by splitting on white spaces, computes unique term frequencies, and exports results to a `.csv` format.

#### 3. Experiment 4.3: TF-IDF Calculation (without NLP toolkits)
Term Frequency-Inverse Document Frequency (TF-IDF) scores how important a word is to a document in a collection or corpus. This experiment manually implements TF-IDF indexing from mathematical first principles.
*   **Mathematical Models Used**:
    *   $$\text{Term Frequency (TF)} = \frac{\text{Count of term } t \text{ in document } d}{\text{Total words in document } d}$$
    *   $$\text{Inverse Document Frequency (IDF)} = \log_{10}\left(\frac{\text{Total number of documents } N}{\text{Document Frequency of term } t}\right)$$
    *   $$\text{TF-IDF} = \text{TF} \times \text{IDF}$$
*   **Implementation**:
    *   Calculates TF, DF, and IDF scores step-by-step for a controlled sample corpus.
    *   Identifies the top 10 terms with the highest TF-IDF scores per document to isolate high-value keywords from background noise.
 

### 🧠 LAB 5: Advanced Subword Tokenization, POS Tagging, & Word Vector Embeddings

This lab deep-dives into advanced vocabulary modeling strategies, syntactic analysis, and dense semantic representations. In modern NLP pipelines, handling out-of-vocabulary (OOV) terms, tagging grammatical structure, and understanding word-to-word relationships are key layers before feeding textual features into deep learning models.

---

### 🔬 Lab Experiments

#### 1. Experiment 5.1: BPE-based Tokenizer
Byte-Pair Encoding (BPE) is a subword tokenization scheme that builds a vocabulary iteratively by pairing characters or character sequences based on their co-occurrence frequency. It prevents Out-Of-Vocabulary (OOV) errors by splitting unseen words into smaller subword blocks.
*   **a. Pretrained BPE Model (GPT-2)**:
    *   *Implementation*: Utilizes Hugging Face's `GPT2Tokenizer` to convert a complex raw text into subwords and displays corresponding vocabulary IDs.
*   **b. Custom BPE from Scratch**:
    *   *Implementation*: Implements the raw token-frequency merging algorithm from scratch, starting with character-level splits and iteratively merging the most frequent byte pairs up to a custom frequency limit.

#### 2. Experiment 5.2: SentencePiece Tokenizer
SentencePiece is a language-independent subword tokenizer that treats the input text as a raw character sequence, including spaces (modeled with meta-symbol `_` or `▁`), removing the hard dependency on language-specific pre-tokenizers.
*   **a. Pretrained SentencePiece (XLNet)**:
    *   *Implementation*: Implements Hugging Face's `XLNetTokenizer` (`xlnet-base-cased`) to tokenize and encode paragraphs into dense SentencePiece vocab units.
*   **b. Custom SentencePiece Training**:
    *   *Implementation*: Trains a fresh unigram SentencePiece model directly on a local corpus using `sentencepiece.SentencePieceTrainer` and processes raw text through the generated vocabulary weights.

#### 3. Experiment 5.3: Part-of-Speech (POS) Tagging on Sample Text
Part-of-Speech tagging assigns grammatical roles (e.g., nouns, verbs, determiners) to words in a sentence, which is crucial for downstream parsing tasks.
*   **a. spaCy Tagging**:
    *   *Implementation*: Analyzes the syntactic structure of a sentence using spaCy's optimized transformer-based neural models (`en_core_web_sm`), exporting granular part-of-speech tags.
*   **b. NLTK Tagging**:
    *   *Implementation*: Evaluates the same sample sentence using standard Penn Treebank tagging algorithms from the Natural Language Toolkit (`nltk`).

#### 4. Experiment 5.4: POS Tagging & Frequency Mapping on Corpus
Grammatical composition varies widely across contexts and corpora.
*   **Implementation**: Parses an entire input document using spaCy, computes the frequencies of each unique word, and prints a structured tabular report mapping tokens, their tags, descriptions, and occurrence frequencies.

#### 5. Experiment 5.5: Word2Vec - Continuous Bag-of-Words (CBOW)
Word2Vec models map words to low-dimensional continuous vector spaces. The CBOW architecture learns to predict a target center word given its surrounding context words.
*   **Implementation**: Trains a 50-dimensional Gensim `Word2Vec` model on a custom 10-line text corpus. It extracts the numerical vector for the word `"learning"` and reports the 3 most similar words based on cosine distance.

#### 6. Experiment 5.6: Word2Vec - Skip-Gram
The Skip-Gram architecture flips CBOW; it uses a single center word to predict surrounding context words, capturing highly specific semantic relationships even for rarer words.
*   **Implementation**: Configures Gensim's `Word2Vec` parameter `sg=1` to train a Skip-Gram network on the corpus, retrieves the dense embedding for `"language"`, and prints its top 5 semantically closest neighbors.

#### 7. Experiment 5.7: Semantic Cosine Similarity Evaluation
Vector mathematics allows us to compute the exact semantic similarity between concepts by calculating the angle between their high-dimensional vectors.
*   **Implementation**: Extracts trained embeddings from the neural layers and evaluates similarity ratings for specific bigram pairings including `("machine", "learning")` and `("natural", "language")` using cosine metric algorithms.
