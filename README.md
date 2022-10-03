
# Information Extraction - Assignment 2

<b>Student Id: 17230755</b>


### Overview:
This assignment is regarding Information extraction in a football player data set. I have used a standford tagger for Named Entity extraction. 

Please Note:

    - Need to set environment to os.environ['JAVAHOME'] = "C:/Program Files/Java/jre1.8.0_144/bin/" (corresponding relative path should be given according to the system)
    - Need to downlaod the standfor NER tagger from the site here - http://nlp.stanford.edu/software/stanford-postagger-full-2015-04-20.zip
    - If this dosent work - please uncomment the code in name_0f_the_player function. Instruction is given in comments.



```python
# Import all necessary tools required
import os
import nltk
import re
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import itertools
from pyld import jsonld
import json
import unidecode
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger('C:/Users/Swaroop Bhat/Anaconda3/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz',
               'C:/Users/Swaroop Bhat/Anaconda3/stanford-ner-2014-06-16/stanford-ner.jar')
os.environ['JAVAHOME'] = "C:/Program Files/Java/jre1.8.0_144/bin/"
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from difflib import SequenceMatcher
```

### Loading the file:

Please make sure txt file is in working directory.


```python
# Please make sure the txt file is in home directory
inputfile='football_players.txt'
buf=open(inputfile, encoding="UTF-8")
list_of_doc=buf.read().split('\n')


# Removing all empty lines i.e empty lines between paragraph
l = []
for i in list_of_doc:
    if len(i) != 0:
        l.append(i)
list_of_doc = l
```

# Task 1 (10 Marks)
Write a function that takes each document and performs:
1) sentence segmentation 2) tokenization 3) part-of-speech tagging

Please keep in mind that the expected output is a list within a list as shown below.

All abobve mentioned process is done.


```python
def ie_preprocess(document):
    
    '''This function pos tag the sentences and stores in the list'''
    
    # Convert string into list of sentences
    sentence = sent_tokenize(document)
    pos_sentences = []
    try:
        for i in sentence:
            # Tokenize each sentence
            text = word_tokenize(i)
            # Tag the tokenizeed sentence
            pos_sentences.append(nltk.pos_tag(text))
        return pos_sentences
    except:
        print("Please make sure that input to the function is of type string")
```


```python
first_doc=list_of_doc[0]
# Calling the ie_preprocess function
pos_sent = ie_preprocess(first_doc)

# Display all tagged sentences as given in the assignment sheet. However, to dissplay all tagged sentence, please remove the 
# list index
pos_sent[1]
```




    [('He', 'PRP'),
     ('is', 'VBZ'),
     ('a', 'DT'),
     ('forward', 'NN'),
     ('and', 'CC'),
     ('serves', 'NNS'),
     ('as', 'IN'),
     ('captain', 'NN'),
     ('for', 'IN'),
     ('Portugal', 'NNP'),
     ('.', '.')]



# Task 2 (20 Marks)
Write a function that will take the list of tokens with POS tags for each sentence and returns the named entities (NE). 

Hint: Use binary=True while calling NE chunk function.


```python
def named_entity_finding(pos_sent, x=True, y="NE"):
    
    """This functions identifies the named entities in a given pos tagged sentence. This logic is used in another 
    function to change binary=True in ne_chunk. Hence given the default parameters in the function. 
    Also using this we can find the NE labeled - person or location... etc default is person.""" 
    
    # This is check if the return needs all NE or specific NE - default type is PERSON.
    if y != "NE":
        x=False
        if y=="NE":
            y = "PERSON"
    myNE = []
    
    
    # The below logic finds all named entities like, PERSON, ORGANIZATION, LOCATION etc
    try:
        tree = nltk.ne_chunk(pos_sent, binary=x)

        for subtree in tree.subtrees():
            if subtree.label() == y:
                entity = ""
                for leaf in subtree.leaves():
                    entity = entity + leaf[0] + " "
                myNE.append(entity.strip())

        return myNE
    except:
        print("Please make sure to give parsed tree as input to the function.")

        
# Calling the above function - specific document need to be passed.
pos_sents=ie_preprocess(list_of_doc[0])
# passing the first sentence of document 4 to findd nam
named_entity_finding(pos_sents[0])
```




    ['Cristiano Ronaldo',
     'Santos Aveiro',
     'ComM',
     'GOIH',
     'Portuguese',
     'Spanish',
     'Real Madrid',
     'Portugal']



# Task 3 (10 Marks)

Now use the named_entity_finding() function to extract all NEs for each document.

Hint: pos_sents holds the list of lists of tokens with POS tags.

Note: Only few list of NE is displayed.


```python
def NE_flat_list_fn(document):
    """This function is extracts named entities in entire document and returns all unique NE as flattened list"""
    NE=[]
    try:
        # Loop through each docudment 1-10
        for i in document:
            # POS tag the document
            pos_sents = ie_preprocess(i)
            for pos_sent in pos_sents:
                # For each tagged sentence in document find named entity
                entity = named_entity_finding(pos_sent)
                if len(entity) != 0:
                    NE.append(entity)

        
        # This function flattens the list
        NE_flat_list = list(itertools.chain.from_iterable(NE))
        return NE_flat_list
    except:     
        print("Please make sure that input to the function is off type string")

        
ne_flat = NE_flat_list_fn(list_of_doc)

# Note set is used to return the unique named entities - only 10 named entites are displayed here. 
# There are multiple same named entities - However, for any process - if we take the unique NE it is better in tems of memory. 
# This set can also be declared instind the NE_flat_list_n.
list(set(ne_flat))[1:10]
```




    ['UEFA European',
     'British',
     'FIFA Puskás Award',
     'Spain',
     'German',
     'Dagens Nyheter',
     'Ronaldo',
     'Marco',
     'OBE']



# Task 4 (40 Marks)

Write functions to extract the name of the player, country of origin and date of birth as well as the following relations: team(s) of the player and position(s) of the player.

Hint: Use the re.compile() function to create the extraction patterns

Reference: https://docs.python.org/3/howto/regex.html


```python
def name_of_the_player(doc):
    
    """This function finds the person name"""
    
    t = []
    name = []
    try:
        pos_sents = ie_preprocess(doc)
        tree = nltk.ne_chunk(pos_sents[0])
        
        
        """ This is to extract player name using standfor NE tagger. Standford works slightly better than NLTK default tagger.
        Moreover, Standfor NE tagger detects the surname as well.
        E.g: NLTK default - Cristano Ronaldo : retuns Crisstano as PERSON and Ronaldo as ORGANIZATION. Hence using standford package."""
        
        sentence = unidecode.unidecode(sent_tokenize(doc)[0])
        tag = st.tag(sentence.split())
        for i in tag:
            if i[1] == "PERSON":
                t.append(i[0])
        
        return " ".join(t)

        
        """Note: If not able to install standfor NE tagger - please uncomment the below section (while commenting above block)"""
        
        #for subtree in tree.subtrees():
            #if subtree.label() == "PERSON":
                #entity = ""
                #for leaf in subtree.leaves():
                    #entity = entity + leaf[0] + " "
                #name.append(entity.strip())

        #return " ".join(name)
    
    except:
        print("Please make sure the entered documnent is of type string.")


try:
    print(name_of_the_player(list_of_doc[0]))
except:
    print("Please make sure give list index is right")
```

    Cristiano Ronaldo dos Santos
    


```python
def country_of_origin(doc):
    
    """ Three levels of filter is applied here. Because NLTK NE tagger dosent recognise all the country names. Moreover, 
    we can use look up to get the country name (because the text contains like spanish that can be converted to Spain).
    LEVEL 1 - Used NLTK tagger to find Named Entity. If not found go to LEVEL - 2
    LEVEL 2 - Used Regex to check for word with profesional footballer. If not found go to LEVEL - 3
    LEVEL 3 - USed Regex to match word with National Team.
    Note - Not used Look up here. This type of Levels is useful in a large documents"""
    
    sentence = sent_tokenize(doc)[0]
    tm = []
    seq = []
    
    try:   
        # LEVEL 2 (look for prrofesssional footballer.)
        # This uses regex to find the country of origin if NLTK fails to detect GPE or LOCATION tag.
        m = re.search(r'((?:\w+\W+){,3})(footballer)', sentence)
        s = "".join(list(m.groups()))
        mal = re.search(r'((\w+ ){1})professional', s)
        # Take the first group most probably the country name
        try:
            mal = mal.groups()[0]
        except:
            mal = None

        # LEVEL - 1 (Primary lever, get NE for 1st sentence and look for GPE)
        # Get tag for first sentence
        tag = ie_preprocess(s)[0]
        tree = nltk.ne_chunk(tag)
        born = []

        # Check for GPE tag if any (sometimes NLTK dosent detect GPE).
        for subtree in tree.subtrees():
            if subtree.label() == 'GPE':
                entity = ""
                for leaf in subtree.leaves():
                    entity = entity + leaf[0] + " "
                born.append(entity.strip())


        # LEVEL 3 (if nowhere found then check for words with national team.)
        match = re.compile(r'((?:[\S,]+\s+){0,1})national team')
        for i, sent in enumerate(sent_tokenize(doc)):
            if i == 2:
                break
            team = match.findall(sent)
            if len(team) != 0:
                tm.append(team[0])


        # If NLTK fails to detect then use Regex re.search(r'((?:\w+\W+){,3})(footballer)', sentence) - i.e take 3 words before 
        # the word footballer and take the first group.
        if len(born) == 0:
            if len(mal) != 0:
                born = nltk.word_tokenize(mal) 
            elif len(tm) != 0:
                born = tm
        
        # Lookup of all the countries - can use all country name as look up but included only few.
        # sequence matcher is used to find the country in look up.
        l = ['Spain', 'England', 'Portugal', 'Wales', 'England', 'Brazil', 'Germany', 'Argentina', 'Sweedan']
        for x in l:
            seq.append(SequenceMatcher(None, born[0], x).ratio())
        final = l[seq.index(max(seq))]
            
        
        return final
    except:
        print("Please make sure the data given to this function is of type string")

try:
    print(country_of_origin(list_of_doc[0]))
except:
    print("Please make sure the given index is right!!")
```

    Portugal
    


```python
def date_of_birth(doc):
    
    """Used regex to find the date of bith. First extract the sentence containing born and find the matching pattern."""
    
    sentence = sent_tokenize(doc)[0]
    match = re.compile(r'born\b\s*((?:\S+\s+){0,3})')
    born = match.findall(sentence)[0]
    born = re.sub('\W+',' ', born )
    
    return born


try:
    print(date_of_birth(list_of_doc[2]))
except:
    print("Please make sure the list indexing is right")
```

    5 February 1992 
    


```python
def team_of_the_player(doc):
    
    """This functions returns teams of the player. Note Some of the players dosent have national team information
    (for sentence 4). Hence, assumed as player plays for his country of origin (This may not be true for always). """
    
    sentence = sent_tokenize(doc)
    tm = []
    club = []
    club2 = []
    
    try:
        tag = ie_preprocess(doc)
        # Check for only first 2 sentences. Saves time and efficent.
        for i, sent in enumerate(sentence):
            if i == 2:
                break
            # Extract the national team of the player using below regex.
            match = re.compile(r'((?:[\S,]+\s+){0,1})national team')
            
            # This is to extract the clubs of the player.
            match_club = re.compile(r'club\s+((?:[\S,]+\s*){0,2})')
            
            # Find the national team
            team = match.findall(sent)
            
            # If club keyword found in first couple of sentences. Extract those clubs.
            if len(match_club.findall(sent)) != 0:
                for f in match_club.findall(sent):
                    club2.append(f)
            
            # Search of clubs. This can be found by checking for organization by calling Named Entity function and type is 
            # ORGANIZATION.
            club.append(named_entity_finding(tag[i], False, "ORGANIZATION"))
            if len(team) != 0:
                l = team[0]+"national team"
                tm.append(l)

        # Find unique names because sentence may have same names multiple times.
        if len(list(set(tm))) != 0:
            national_team = list(set(tm))[0]
        else:
            national_team = country_of_origin(doc) +" national team"

        # Flattens the list
        club = list(itertools.chain.from_iterable(club))
        
        # Finding the most probable club names by doing intersection of two process.
        nclub = []
        for q in club2:
            nclub.extend(nltk.word_tokenize(q))

        for i, s in enumerate(club2):
            club2[i] = club2[i].rstrip()
        if len(list(set(club).intersection(club2))) != 0:
            club = list(set(club).intersection(club2))
        club.append(national_team)


        club = list(set(club).difference(nltk.word_tokenize(sentence[0])[0:6]))

        return club
    except:
        print("Please make sure the data given to this function is of type string")


try:
    print(team_of_the_player(list_of_doc[9]))
except:
    print("Please make sure the given list index is right.")
```

    ['Spain national team', 'FC Barcelona']
    


```python
def position_of_the_player(doc):
    
    """ This function extracts position of the player. Can return multiple values if the player played in 
    different clubs or teams as different position. Hence the return type is list."""


    # This is look up of all postions available in football
    pos = ["forward", "captian", "attacking midfielder", "striker", "winger", "central midfielder", "defensive tackle", "defensive end"]
    player_position = []
    try:
        sent = sent_tokenize(doc)
        for i, sent in enumerate(sent):
            for x in pos:
                # Find matching in look up and sentence.
                regex = re.compile(r'\b({0})\b'.format(x), flags=re.IGNORECASE)
                
                # if result is true append to a list
                r = bool(regex.search(sent))
                
                if r == True:
                    player_position.append(x)
        
        return list(set(player_position))
    except:
        print("Please make sure that input to the function is string")

try:
    print(position_of_the_player(list_of_doc[0]))
except:
    print("Maximum length of document is 10 i.e. index 9")
```

    ['forward']
    

# Task 5 - Json-ld

Here, first we need to generate the data, because arguments is given as list to generate_jsonld function. I have provided with two solution.
    
<b>Solution 1:</b> Simple store the required format in a variable as give the arguments where required. However, not sure wheteher the return type needed is linked data format. Hence given another solution using pyld to get Json ld.
    
<b>Solution 2:</b> Used Pyld to generated Json ld. It contains context and doc - to use this data as linked data then contex and doc is necessay. However, in given format in assignment dosent contain context section. Hence poping context from json ld (not sure if this is allowed - linked data needs context).



Below is the data generaton function to convert all arguments to list.


```python
def data_generator(doc):
    
    data = [name_of_the_player(doc), date_of_birth(doc), country_of_origin(doc), position_of_the_player(doc), team_of_the_player(doc)]
    
    return data


data = data_generator(list_of_doc[0])
print(data)
```

    ['Cristiano Ronaldo dos Santos', '5 February 1985 ', 'Portugal', ['forward'], ['Real Madrid', 'Portugal national team']]
    

### Solution 1:


```python
def generate_jsonld1(arg, con=True):
    if con == True:

        ld = { "@id": "http://my-soccer-ontology.com/footballer/"+arg[0],

            "name": arg[0],
            "born": arg[1],
            "country": arg[2],
            "position": [
                { "@id": "http://my-soccer-ontology.com/position/",
                    "type": arg[3]
                }
             ],   
             "team": [
                { "@id": "http://my-soccer-ontology.com/team/",
                    "name": arg[4]
                }   
             ]
        }

        return json.dumps(ld)
    
    elif con == False:
        
        ld = { "@id": "http://my-soccer-ontology.com/footballer/"+arg[0],

            "name": arg[0],
            "born": arg[1],
            "country": arg[2],
            "position": [
                { "@id": "http://my-soccer-ontology.com/position",
                    "type": arg[3]
                }
             ],   
             "team": [
                { "@id": "http://my-soccer-ontology.com/team",
                    "name": arg[4]
                },
             ],
            "Debut Year": arg[5][0],
            "Debut Age": arg[5][1]
        }

        return json.dumps(ld)
        


try:
    data = data_generator(list_of_doc[0])
    print(generate_jsonld1(data))
except:
    print("Please make sure that list index given is right")

```

    {"@id": "http://my-soccer-ontology.com/footballer/Cristiano Ronaldo dos Santos", "name": "Cristiano Ronaldo dos Santos", "born": "5 February 1985 ", "country": "Portugal", "position": [{"@id": "http://my-soccer-ontology.com/position/", "type": ["forward"]}], "team": [{"@id": "http://my-soccer-ontology.com/team/", "name": ["Real Madrid", "Portugal national team"]}]}
    

### Solution 2:


```python
def generate_jsonld2(arg, con=True):
    
    if con == True:
        doc = {

            "http://my-soccer-ontology.com/footballer/name_of_the_player":  
            "http://my-soccer-ontology.com/footballer/" +arg[0],
            "http://my-soccer-ontology.com/name": arg[0],
            "http://my-soccer-ontology.com/born": arg[1],
            "http://my-soccer-ontology.com/country": arg[2],
            "http://my-soccer-ontology.com/team": {"@id": "http://my-soccer-ontology.com/team", "@type": arg[4]},
            "http://my-soccer-ontology.com/position": {"@id": "http://my-soccer-ontology.com/position", "@type": arg[3]}
        }

        context = {
            " @id": "http://my-soccer-ontology.com/footballer/name_of_the_player", "name": "http://my-soccer-ontology.com/name",
            "born": "http://my-soccer-ontology.com/born", "country": "http://my-soccer-ontology.com/country",
            "team": {"@id": "http://my-soccer-ontology.com/team", "@type": "@id"}, 
            "position": {"@id": "http://my-soccer-ontology.com/position", "@type": "@id"}
        }
        
        compacted = jsonld.compact(doc, context)
        
        # Data preprocess to look like the format given in assignment question.
        # Pop the context and renaming several attributes:
        compacted.pop('@context')
        compacted['position']['type'] = compacted['position'].pop('@type')
        compacted['team']['name'] = compacted['team'].pop('@type')
        
        # Cleaning the text containing '/' character
        if type(compacted['position']['type']) == str:
            compacted['position']['type'] = compacted['position']['type'].replace("/", "")
        else:
            compacted['position']['type'] = [position.replace("/", "") for position in compacted['position']['type']]
        
        if type(compacted['team']['name']) == str:
            compacted['team']['name'] = compacted['team']['name'].replace("/", "")
        else:
            compacted['team']['name'] = [name.replace("/", "") for name in compacted['team']['name']]
        
        return compacted
    
    if con == False:
            
        doc = {

            "http://my-soccer-ontology.com/footballer/name_of_the_player":  
            "http://my-soccer-ontology.com/footballer/" +arg[0],
            "http://my-soccer-ontology.com/name": arg[0],
            "http://my-soccer-ontology.com/born": arg[1],
            "http://my-soccer-ontology.com/country": arg[2],
            "http://my-soccer-ontology.com/debutYear": arg[5][0],
            "http://my-soccer-ontology.com/debutAge": arg[5][1],
            "http://my-soccer-ontology.com/team": {"@id": "http://my-soccer-ontology.com/team", "@type": arg[4]},
            "http://my-soccer-ontology.com/position": {"@id": "http://my-soccer-ontology.com/position", "@type": arg[3]}
        }

        context = {
            " @id": "http://my-soccer-ontology.com/footballer/name_of_the_player", "name": "http://my-soccer-ontology.com/name",
            "born": "http://my-soccer-ontology.com/born", "country": "http://my-soccer-ontology.com/country",
            "debutYear": "http://my-soccer-ontology.com/debutYear", "debutAge":"http://my-soccer-ontology.com/debutAge",
            "team": {"@id": "http://my-soccer-ontology.com/team", "@type": "@id"}, 
            "position": {"@id": "http://my-soccer-ontology.com/position", "@type": "@id"}
        }
        
        compacted = jsonld.compact(doc, context)
        
        # Data preprocess to look like the format given in assignment question.
        # Pop the context and renaming several attributes:
        compacted.pop('@context')
        compacted['position']['type'] = compacted['position'].pop('@type')
        compacted['team']['name'] = compacted['team'].pop('@type')
        
        # Cleaning the text containing '/' character
        if type(compacted['position']['type']) == str:
            compacted['position']['type'] = compacted['position']['type'].replace("/", "")
        else:
            compacted['position']['type'] = [position.replace("/", "") for position in compacted['position']['type']]
        
        if type(compacted['team']['name']) == str:
            compacted['team']['name'] = compacted['team']['name'].replace("/", "")
        else:
            compacted['team']['name'] = [name.replace("/", "") for name in compacted['team']['name']]
        
        return compacted

    
data = data_generator(list_of_doc[0])
print(generate_jsonld2(data))
```

    {'born': '5 February 1985 ', 'country': 'Portugal', ' @id': 'http://my-soccer-ontology.com/footballer/Cristiano Ronaldo dos Santos', 'name': 'Cristiano Ronaldo dos Santos', 'position': {'@id': 'http://my-soccer-ontology.com/position', 'type': 'forward'}, 'team': {'@id': 'http://my-soccer-ontology.com/team', 'name': ['Real Madrid', 'Portugal national team']}}
    

# Task 6 - Debut Year and Debut Age

I have take debut year and debut age of the player. However, it is important to note that, these two values are not available for all the players. Hence, if debut year or debut age is not present I have added as "Not Available".


```python
def relation_debutYearAge(doc):
    
    """This function extracts the debut year and debut age of the player. 
    Note - this information if not available for all the players"""
    
    sent = sent_tokenize(doc)
    deb = []
    ag = []
    # For each sentence check if the debut key is present or not
    for se in sent:
        sp_sent = se.split()
        
        if "debut" in sp_sent:
            # If debut year is present jecy for 4 digit number.
            date = re.findall('\d{4}', " ".join(sp_sent))
            if len(date) !=0:
                deb.append(date[0])
        
        # Check for debut and age key in a sentence - 
        if "debut" in sp_sent and "aged" in sp_sent:
            # Search for 2 digit number and append to the list.
            age = re.findall('\d{2}', " ".join(sp_sent))
            ag.append(age[0])       
    
    # If the above process dosent return anything, insert year and age is not available.
    if len(deb) == 0:
        deb.append("Not Available")
    if len(ag) == 0:
        ag.append("Not Available")

    return [deb[0], ag[0]]

relation_debutYearAge(list_of_doc[1])


# Note: Debut year or debut age is not available for all the sntences (3,7). Moreover, some of the sentence has only
# debut age, while the others has only debut year.
```




    ['2004', '17']




```python
def data_generator(doc):
    
    data = [name_of_the_player(doc), date_of_birth(doc), country_of_origin(doc), position_of_the_player(doc), team_of_the_player(doc), relation_debutYearAge(doc)]
    return data


data = data_generator(list_of_doc[0])
print(data)
```

    ['Cristiano Ronaldo dos Santos', '5 February 1985 ', 'Portugal', ['forward'], ['Real Madrid', 'Portugal national team'], ['2003', 'Not Available']]
    

### Json-Ld from solution1:


```python
generate_jsonld1(data, False)
```




    '{"@id": "http://my-soccer-ontology.com/footballer/Cristiano Ronaldo dos Santos", "name": "Cristiano Ronaldo dos Santos", "born": "5 February 1985 ", "country": "Portugal", "position": [{"@id": "http://my-soccer-ontology.com/position", "type": ["forward"]}], "team": [{"@id": "http://my-soccer-ontology.com/team", "name": ["Real Madrid", "Portugal national team"]}], "Debut Year": "2003", "Debut Age": "Not Available"}'



### Json-Ld from solution2:


```python
generate_jsonld2(data, False)
```




    {' @id': 'http://my-soccer-ontology.com/footballer/Cristiano Ronaldo dos Santos',
     'born': '5 February 1985 ',
     'country': 'Portugal',
     'debutAge': 'Not Available',
     'debutYear': '2003',
     'name': 'Cristiano Ronaldo dos Santos',
     'position': {'@id': 'http://my-soccer-ontology.com/position',
      'type': 'forward'},
     'team': {'@id': 'http://my-soccer-ontology.com/team',
      'name': ['Real Madrid', 'Portugal national team']}}


