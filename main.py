from agents import DeepResearchAgent,ReviewAgent,get_llms
import asyncio
import json
import argparse
import yaml
import os
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
for key, value in config.items():
    if value == "":
        continue
    else:
        os.environ[key] = str(value)

LLM_MERGING = """Develop a novel and effective LLM merging method to improve performance on held out test set within the time constraints.

## Description
Training high-performing large language models (LLMs) from scratch is a notoriously expensive and difficult task, costing hundreds of millions of dollars in compute alone. These pretrained LLMs, however, can cheaply and easily be adapted to new tasks via fine-tuning, leading to a proliferation of models that suit specific use cases. Recent work has shown that specialized fine-tuned models can be rapidly merged to combine capabilities and generalize to new skills.

The competition will provide the participants with a list of expert models that have already been trained on a task-specific dataset. The goal of this competition is to re-use the provided models to create a generalist model that can perform well on a wide variety of skills like reasoning, coding, maths, chat, and tool use. Along with these expert models, we have a set of hidden tasks that will be used to evaluate the submissions from participants."""

BACKDOOR_TRIGGER_RECOVERY = """**Backdoor Trigger Recovery for Code Generation Models**

## Description
Participants in this competition are tasked with developing algorithms to recover backdoor triggers embedded within large language models (LLMs) used for code generation. Each provided backdoored LLM contains multiple (trigger, target) pairs, where triggers are universal prompt injections designed to induce the generation of malicious code specified by the targets. In the development phase, participants receive a model finetuned with five known (trigger, target) pairs, while in the testing phase, the models include tens of secret (trigger, target) pairs related to various categories of harmful code generation. The objective is to predict the triggers corresponding to each provided target, adhering to a maximum token constraint of 10 tokens per trigger. Submissions will be evaluated using two metrics: recall, which measures the similarity between predicted and ground truth triggers, and the Reverse-Engineering Attack Success Rate (REASR), which assesses the effectiveness of the recovered triggers in eliciting the malicious code. Participants are provided with a starter dataset of 100 code generation queries and their correct outputs for method development and local evaluation, with additional data encouraged for enhancing method robustness. However, any attempts to access or guess the secret online evaluation dataset will be considered a rule violation."""

TASK2ANCHOR = {
        "llm-merging" : "dare",
        "backdoor-trigger-recovery" : "gcg",
        # TODO: add one paper name for each task
        }

TASK2TOPIC = {
        "llm-merging" : LLM_MERGING,
        "backdoor-trigger-recovery" : BACKDOOR_TRIGGER_RECOVERY,
        # TODO: add task description
        }


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # argparser.add_argument("--topic",type=str,help="research topic",default="Using diffusion to generate urban road layout map")
    argparser.add_argument("--task",type=str,help="research task name")
    argparser.add_argument("--anchor_paper_path",type=str,help="PDF path of the anchor paper",default= None)
    argparser.add_argument("--save_file",type=str,default="saves/",help="save file path")
    argparser.add_argument("--improve_cnt",type=int,default=1,help="experiment refine count")
    argparser.add_argument("--max_chain_length",type=int,default=10,help="max chain length")
    argparser.add_argument("--min_chain_length",type=int,default=5,help="min chain length")
    argparser.add_argument("--max_chain_numbers",type=int,default=1,help="max chain numbers, not used if anchor_paper_path is not None")
    argparser.add_argument("--idea_idx",type=int,default=None,help="index for ideas if multiple ideas are to be sampled")
    
    args = argparser.parse_args()

    main_llm , cheap_llm = get_llms()
    task = args.task

    topic = TASK2TOPIC[task]
    anchor_paper_path = args.anchor_paper_path
    if anchor_paper_path is None:
        anchor_paper_path = f"papers/{TASK2ANCHOR[task]}.pdf"


    review_agent = ReviewAgent(save_file=args.save_file,llm=main_llm,cheap_llm=cheap_llm)
    deep_research_agent = DeepResearchAgent(llm=main_llm,cheap_llm=cheap_llm,**vars(args))

    # print(f"begin to generate idea and experiment of topic {topic}")
    idea,related_experiments,entities,idea_chain,ideas,trend,future,human,year=  asyncio.run(deep_research_agent.generate_idea_with_chain(topic,anchor_paper_path))
    """
    experiment = asyncio.run(deep_research_agent.generate_experiment(idea,related_experiments,entities))

    for i in range(args.improve_cnt):
        experiment = asyncio.run(deep_research_agent.improve_experiment(review_agent,idea,experiment,entities))
    """ 
    # print("skip generating experiment")
    experiment = ""
    # print(f"succeed to generate idea and experiment of topic {topic}")
    api_cost = main_llm.api_cost + cheap_llm.api_cost  
    res = {"api_cost" : api_cost, "idea":idea,"experiment":experiment,"related_experiments":related_experiments,"entities":entities,"idea_chain":idea_chain,"ideas":ideas,"trend":trend,"future":future,"year":year,"human":human}
    anchor_paper_name = os.path.basename(anchor_paper_path).replace(".pdf", "")
    outfile_dir = os.path.join("results", task, os.environ['MAIN_LLM_MODEL'])
    if args.idea_idx is not None:
        outfile_dir = os.path.join(outfile_dir, f"{args.idea_idx}")
    os.makedirs(outfile_dir, exist_ok=True)
    outfile = os.path.join(outfile_dir, "result.json")
    with open(outfile,"w") as f:
        json.dump(res,f, indent=2)
    print(f"Idea saved to {outfile}")
