from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = stopwords.words('english')


from textblob import Word

#experimenting with lemmatisation
def check_word_spelling(word):

    word = Word(word)

    result = word.spellcheck()

    if word == result[0][0]:
        print(f'Spelling of "{word}" is correct!')
    else:
        print(f'Spelling of "{word}" is not correct!')
        print(f'Correct spelling of "{word}": "{result[0][0]}" (with {result[0][1]} confidence).')


#check_word_spelling('appple ')

def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)

sentence = "The striped bats are hanging on their feet for best"
#print(lemmatize_with_postag(sentence))

from textblob import TextBlob

phrase = "the boook wass horible"
tb_phrase = TextBlob(phrase)
print(tb_phrase.correct())
correct = tb_phrase.correct()
print(correct)
tb_correct = TextBlob(str(correct))
print(tb_correct.tags)
print(tb_correct.sentiment)

stripped_phrase = []
phrase = "The striped bats are hanging on their feet for best"
words = word_tokenize(phrase) 
filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)
#stripped_phrase.append(word)
#" ".join(stripped_phrase)

stemmer = PorterStemmer()

phrase = "reading the book"
words = word_tokenize(phrase)
stemmed_words = []
for word in words:
    stemmed_words.append(stemmer.stem(word))

" ".join(stemmed_words)
