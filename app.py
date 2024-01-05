import os
import typer
import warnings

from transformers.utils.logging import set_verbosity_error, disable_progress_bar
from huggingface_hub import try_to_load_from_cache

from rich import print
from rich.console import Console
from rich.table import Table

import database as db
from pipeline import Pipeline, Node
from tqdm import tqdm


app = typer.Typer()
MODEL_MANAGER = None

def build_from_dict_topic_classification():
    import json
    with open('sentences_for_topic_classification.json') as f:
        samples = json.load(f)
    with open('template_topic_classification.json') as f:
        parsed_json = json.load(f)

    pipe_line = Pipeline()

    nodes = [Node(**node) for node in parsed_json['data']['nodes']]
    for node in nodes:
        pipe_line.add_node(node)

    for edge in parsed_json['data']['edges']:
        pipe_line.add_edge(**edge)

    print("Visualize graph:")
    pipe_line.draw()
    print("\n\n\n")

    print("Topologically sorted nodes:")
    sorted_nodes = pipe_line.topological_sort()
    for node in sorted_nodes:
        print(node.wrapper_class)
    print("\n\n\n")

    for sentence in tqdm(samples):
        for node in sorted_nodes:
            print(node.wrapper_class)
            if node.wrapper_class is ChatTemplateWrapper:
                node.parameters["messages"] = [
                            {"role": "user",
                            "content": f"Step 1: Subject Identification\n\nChoose 1 to 3 relevant categories from the provided list based on the subject and context of the sentence. Each chosen category should be separated by ' / '.\n\nCategories: personal life, personal trivia, home life, basic necessities, school life, friendships, social life, interpersonal relationships, hobbies, leisure time, personal entertainment, travel, animals, plants, seasons, weather, natural phenomena, multicultural family, various cultural backgrounds, linguistic and cultural differences, introducing various Korean cultures and lifestyles, public morals, etiquette, cooperation, consideration, volunteer service, responsibility, environmental conservation, climate change, resources and energy issues, literature, art, aesthetic sensibilities, creativity, imagination, population issues, youth issues, aging issues, multicultural societies, information and communication ethics, career path, jobs and labor, personal welfare, democratic civic life, human rights, gender equality, global etiquette, global citizenship, patriotism, world peace, national security, unification, general knowledge, politics, economics, history, geography, mathematics, science, transportation, information and communication, space, the ocean, and exploration, academic knowledge, humanities, social sciences, natural sciences, digital issues, artificial intelligence, robotics, digital literacy, data science, machine learning.\n\nSentence for Analysis: 'I enjoy taking long walks in the park to clear my mind.'\nSelected Categories: leisure time / personal entertainment\n\nSentence for Analysis: '{sentence}'\nSelected Categories: "}
                        ]
            node.run()


def build_from_dict_syntactic_parser():
    import json
    with open('sentences_for_syntactic_parser.json') as f:
        samples = json.load(f)
    with open('template_syntactic_parser.json') as f:
        parsed_json = json.load(f)

    pipe_line = Pipeline()

    nodes = [Node(**node) for node in parsed_json['data']['nodes']]
    for node in nodes:
        pipe_line.add_node(node)

    for edge in parsed_json['data']['edges']:
        pipe_line.add_edge(**edge)

    print("Visualize graph:")
    pipe_line.draw()
    print("\n\n\n")

    print("Topologically sorted nodes:")
    sorted_nodes = pipe_line.topological_sort()
    for node in sorted_nodes:
        print(node.wrapper_class)
    print("\n\n\n")

    prompt = f"""
The following is the Penn Treebank POS Tagset:
1.	CC	Coordinating conjunction
2.	CD	Cardinal number
3.	DT	Determiner
4.	EX	Existential there
5.	FW	Foreign word
6.	IN	Preposition or subordinating conjunction
7.	JJ	Adjective
8.	JJR	Adjective, comparative
9.	JJS	Adjective, superlative
10.	LS	List item marker
11.	MD	Modal
12.	NN	Noun, singular or mass
13.	NNS	Noun, plural
14.	NNP	Proper noun, singular
15.	NNPS	Proper noun, plural
16.	PDT	Predeterminer
17.	POS	Possessive ending
18.	PRP	Personal pronoun
19.	PRP$	Possessive pronoun
20.	RB	Adverb
21.	RBR	Adverb, comparative
22.	RBS	Adverb, superlative
23.	RP	Particle
24.	SYM	Symbol
25.	TO	to
26.	UH	Interjection
27.	VB	Verb, base form
28.	VBD	Verb, past tense
29.	VBG	Verb, gerund or present participle
30.	VBN	Verb, past participle
31.	VBP	Verb, non-3rd person singular present
32.	VBZ	Verb, 3rd person singular present
33.	WDT	Wh-determiner
34.	WP	Wh-pronoun
35.	WP$	Possessive wh-pronoun
36.	WRB	Wh-adverb

The following is the Penn Treebank Syntactic Tagset:
Number Tag Description
1.  ADJP     Adjective phrase
2.  ADVP     Adverb phrase
3.  NP       Noun phrase
4.  PP       Prepositional phrase
5.  S        Simple declarative clause
6.  SBAR     Clause introduced by subordinating conjunction or 0
7.  SBARQ    Direct question introduced by wh-word or wh-phrase
8.  SINV     Declarative sentence with subject-aux inversion
9.  SQ       Subconstituent of SBARQ excluding wh-word or wh-phrase
10. VP       Verb phrase
11. WHADVP   Wh-adverb phrase
12. WHNP     Wh-noun phrase
13. WHPP     Wh-prepositional phrase
14. X        Constituent of unknown or uncertain category

The following is a bracketing output sample:
( (S
    (NP (NBAR (ADJP (ADJ "Battle-tested/JJ")
                    (ADJ "industrial/JJ"))
              (NPL "managers/NNS")))
    (? (ADV "here/RB"))
    (? (ADV "always/RB"))
    (AUX (TNS *))
    (VP (VPRES "buck/VBP")))
    (? (PP (PREP "up/RP")
           (NP (NBAR (ADJ "nervous/JJ")
                     (NPL "newcomers/NNS")))))
    (? (PP (PREP "with/IN")
           (NP (DART "the/DT")
               (NBAR (N "tale/NN"))
                     (PP of/PREP
                         (NP (DART "the/DT")
                             (NBAR (ADJP
                                   (ADJ "first/JJ"))))))))
    (? (PP of/PREP
           (NP (PROS "their/PP$")
               (NBAR (NPL "countrymen/NNS"))))
    (? (S (NP (PRO *))
              (AUX to/TNS)
              (VP (V "visit/VB")
                  (NP (PNP "Mexico/NNP")))))
    (? (MID ",/,"))
    (? (NP (IART "a/DT")
           (NBAR (N "boatload/NN"))
                 (PP of/PREP
                     (NP (NBAR
                         (NPL "warriors/NNS"))))
                 (VP (VPPRT "blown/VBN")
                     (? (ADV "ashore/RB"))
                     (NP (NBAR (CARD "375/CD")
                               (NPL "years/NNS"))))))
    (? (ADV "ago/RB"))
    (? (FIN "./.")))

You are a linguist skilled in analyzing sentence structures. Based on the Penn Treebank POS Tagset and Syntactic Tagset, please perform syntactic parsing on the given sentence and write the result in a bracketing output structure.
Given sentence: {{sentence}}"""

    for sentence in tqdm(samples):
        for node in sorted_nodes:
            print(node.wrapper_class)
            if node.wrapper_class is ChatTemplateWrapper:
                node.parameters["messages"] = [
                            {
                                "role": "user",
                                "content": prompt.format(sentence=sentence)
                            }
                        ]
            node.run()


def build_from_dict():
    import json
    with open('template.json') as f:
        parsed_json = json.load(f)
    pipe_line = Pipeline()

    nodes = [Node(**node) for node in parsed_json['data']['nodes']]
    for node in nodes:
        pipe_line.add_node(node)

    for edge in parsed_json['data']['edges']:
        pipe_line.add_edge(**edge)

    print("Visualize graph:")
    pipe_line.draw()
    print("\n\n\n")

    print("Topologically sorted nodes:")
    sorted_nodes = pipe_line.topological_sort()
    for node in sorted_nodes:
        print(node.wrapper_class)
    print("\n\n\n")

    print("first loop")
    for node in sorted_nodes:
        print(node.wrapper_class)
        node.run()
    print("\n\n\n")

    print("second loop")
    for node in sorted_nodes:
        print(node.wrapper_class)
        node.run()
    print("\n\n\n")

    print("third loop")
    for node in sorted_nodes:
        print(node.wrapper_class)
        if node.wrapper_class is HfBitsAndBytesConfigWrapper:
            node.parameters["load_in"] = "8bit"
        node.run()
    print("\n\n\n")


def build(base_model_id):
    pipe_line = Pipeline()

    bnb_node = Node('bnb1', "HfBitsAndBytesConfig",  properties={"load_in": "4bit"})
    model_node = Node('model1', "HfAutoModelForCasualLM",  properties={"base_model_id": base_model_id})
    tokenizer_node = Node('tok1', "HfAutoTokenizer",  properties={"base_model_id": base_model_id})
    text_generator_node = Node('tg1', "HfModelGenerator", properties={
        "temperature": 0.0,
        "max_new_tokens": 1024,
        "repetition_penalty": 0.0,
        "prompt": "hello my dear"
    })

    pipe_line.add_node(bnb_node)
    pipe_line.add_node(model_node)
    pipe_line.add_node(tokenizer_node)
    pipe_line.add_node(text_generator_node)

    pipe_line.add_edge('bnb1', 'model1', "quantization_config",)
    pipe_line.add_edge('model1', 'tg1', "model")
    pipe_line.add_edge('tok1', 'tg1', "tokenizer",)

    pipe_line.draw()

    sorted_nodes = pipe_line.topological_sort()
    print("Topologically sorted nodes:")
    for node in sorted_nodes:
        print(node.wrapper_class)

    for node in sorted_nodes:
        node.run()

@app.command()
def run_topic_classification():
    global MODEL_MANAGER
    build_from_dict_topic_classification()

@app.command()
def run_syntactic_parser():
    global MODEL_MANAGER
    build_from_dict_syntactic_parser()

@app.command()
def run(base_model_id):
    global MODEL_MANAGER
    name = db.get_model_name_by_id(base_model_id)
    # MODEL_MANAGER = build(name)
    build_from_dict()

@app.command()
def add_model(name: str):
    db.add_model(name)
    typer.echo(f"Model added: {name}")

@app.command()
def list_models():
    rows = db.list_models()

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=12)
    table.add_column("Name", min_width=20)
    table.add_column("Cached", justify="right")

    if rows:
        for row in rows:
            filepath = try_to_load_from_cache(cache_dir=os.path.join(os.environ["HF_HOME"], "hub") ,
                                              repo_id=row[1],
                                              filename="config.json")

            cached_status = "Yes" if isinstance(filepath, str) else "No"
            table.add_row(str(row[0]), row[1], cached_status)
    else:
        console.print("No models found.")
        return

    console.print(table)


if __name__ == "__main__":
    db.initialize_db()
    warnings.filterwarnings('ignore', category=FutureWarning)

    set_verbosity_error()
    # disable_progress_bar()
    app()
