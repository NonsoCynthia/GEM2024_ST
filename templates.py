import os
import json

#just to read the webnlg file
def load_json_data(json_filepath):
    json_data = ""
    try:
        # Open the JSON file in read mode
        with open(json_filepath, 'r') as jsonfile:
          # Read the content of the file
          json_data = json.load(jsonfile)
    except:
        json_string = open(json_filepath).read()
        json_data = json.loads(json_string)

    return json_data


#we'll format the data with some instruction format teplates
#I've varied the instruction for diversity.
#Also creating instructions for both triples to text an d text to triple
#note that we also appended the end of sentence token for the tokenier. important as we'll need to pack
#sevarl samples into a single sample to maximize model seq length and hasten training


#below are the instruction templates we'll use. we will use trl packing fucntion so I' adding eos manually addedd
template0 = """Instruction:\nYou will receive a list of triples containing information about a subject entity. Study the triples carefully and use the information within them to generate a concise text describing key facts about the subject entity. Do not include any information that cannot be directly inferred from the given triples. Your text should be a single coherent paragraph written in proper English.\n\nInput:\n{order_in}\nOutput:\n{text}{tokenizer.eos_token}"""
template1 = """Instruction:\nGiven the triples '{{ENTITY}} | {{ATTRIBUTE}} | {{VALUE}}', generate a coherent sentence that describes the relationship between the entity, its attribute, and the corresponding value.\n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template2 = """Instruction:\nYou will receive a list of triples containing subject, relation and object. Study the triples carefully and use the information within them to generate a concise text describing key facts provided about the subject and object. Do not include any information that cannot be directly inferred from the given triples. Your text should be a single coherent paragraph written in proper English.\n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template3 = """Instruction:\nUse the information contained in the given triples to generate factual, grammatically correct and coherent text in English. Do not go beyond the facts presented in the triples.\n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template4 = """Instruction:\nGenerate factual, grammatically correct and coherent text in English from the given triples.\n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template5 = """Instruction:\nRelying solely onn the facts provided in the given triples, generate factual, grammatically correct and coherent text in English.\n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template6 = """Instruction:\nGenerate factual, grammatically correct and coherent text in English given the provided context.\n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template7 = """Instruction:\nUse the given triples to create sentences that are coherent, grammatically correct and factual. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template8 = """Instruction:\nCreate coherent, grammatically correct and factual text from the given triples. Ensure no ommision and addition beyond the information provided. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template9 = """Instruction:\nAdhere to the facts contained in the triples to generate coherent, grammatically correct and factual text. Ensure no ommision and addition beyond the information provided. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template10 = """Instruction:\nGenerate text from the provided triples. Be creative but also strictly adhere to only the information contained in the given context. Generated text should be coherent, grammatically correct and factual. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template11 = """Instruction:\nUse the given triples to create factual and grammatically correct text in English. Ensue to use all the provided information in the triple in a coherent way.\n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template12 = """Instruction:\nCreate coherent text from the given triples of subject, attribute/relationship and object.  ct text in English.\n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template13 = """Instruction:\nYou will receive a list of triples containing information about a subject entity. Study the triples carefully and use the information within them to generate a concise text describing key facts about the subject entity. Do not include any information that cannot be directly inferred from the given triples. Your text should be a single coherent paragraph written in proper English.\n\nInput:\n{order_in}\n\nGenerated Text:\n{text}{tokenizer.eos_token}"""
template14 = """Instruction:\nGenerate factual, grammatically correct and coherent text in English given the provided context.\n\nInput:\n{order_in}\n\nGenerated Output:\n{text}{tokenizer.eos_token}"""
template15 = """Instruction:\nGenerate text from the provided triples. Be creative but also strictly adhere to only the information contained in the given context. Generated text should be coherent, grammatically correct and factual. \n\nInput:\n{order_in}\n\nText:\n{text}{tokenizer.eos_token}"""
template16 = """Instruction:\nProduce coherent text of one or more sentences using all the facts contained in the given triples. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template22 = """Instruction:\nGenerate factual and coherent text from the following triples. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template23 = """Instruction:\nGenerate factual and coherent text from the following input. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
template24 = """Instruction:\nGenerate factual and coherent text given the following. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""


template17 = """Instruction:\nExtract the set of triples containing {{Subject}}, {{Predicate}} and  {{Object}} or '{{ENTITY}} | {{ATTRIBUTE}} | {{VALUE}} from the given text. \n\nInput:\n{text}\n\nOutput:\n{order_in}{tokenizer.eos_token}"""
template18 = """Instruction:\nGenerate relevant and coherent triples indicating the entity and associated attribute relationships from the given text. \n\nInput:\n{text}\n\nOutput:\n{order_in}{tokenizer.eos_token}"""
template19 = """Instruction:\nYou will be given a passage of text. Your task is to extract all the factual triples present in the text that follow the format:\n\n Subject Relation Object \n\nThe subject and object should be entities/concepts mentioned in the text. The relation describes the relationship between the subject and object. \n\nYour output should be a list of all such triples found in the input text, one per line. If no valid triples can be extracted, output "None". \n\nPay attention to only extract factual, explicit statements from the text. Do not make any inferences or assumptions beyond what is directly stated. \n\nInput:\n{text}\n\nOutput:\n{order_in}{tokenizer.eos_token}"""
template20 = """Instruction:\nInstructions: Extract factual triples from the given text in the format "Subject Relation Object". The subject and object should be entities/concepts mentioned in the text, and the relation describes their relationship. List one triple per line. If no valid triples can be extracted, output "None". \nOnly extract explicit statements from the text, do not make inferences or assumptions. \n\nText:\n{text}\n\nTriples:\n{order_in}{tokenizer.eos_token}"""
template21 = """Task:\nTask: Extract factual triples from the given text passage in the format "Subject Relation Object". \nInstructions:\n\nThe subject and object should be entities or concepts explicitly mentioned in the text. \nThe relation describes the relationship between the subject and object entities. \nList one triple per line in your output. \nIf no valid triples can be extracted, output "None". \nOnly extract factual statements directly from the text. Do not make any inferences or assumptions beyond what is stated. \n\nText:\n{text}\n\nYour Output:\n{order_in}{tokenizer.eos_token}"""

templates = [template0, template1, template2, template3, template4, template5, template6, template7, template8, template9, template10, template11, template12, template13, template14, template15, template16, template22, template23, template24]
inverse_templates = [template17, template18, template19, template20, template21]

