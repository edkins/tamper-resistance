import argparse
import json
import re

def fmt(string):
    return string.replace('\n', '\n        ')

def dsname(path):
    return path.split(',')[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("section", type=str, choices=["retain","adversary","meta"])
    args = parser.parse_args()
    section = args.section

    with open(args.filename, "r") as f:
        data = json.load(f)
    
    if section == 'retain':
        for path, data_dict in data.items():
            ds = dsname(path)
            if section in data_dict:
                for inp in data_dict[section]['input_ids']:
                    print(ds, fmt(inp))
    elif section in ['adversary', 'meta']:
        re_junk = re.compile(r'<\|im_end x \d+\|>')
        for path, data_dict in data.items():
            ds = dsname(path)
            if section in data_dict:
                for prompt, chosen, rejected in zip(data_dict[section]['prompt_input_ids'], data_dict[section]['chosen_input_ids'], data_dict[section]['rejected_input_ids']):
                    prompt = re_junk.sub('', prompt)
                    chosen = re_junk.sub('', chosen)
                    rejected = re_junk.sub('', rejected)
                    # if chosen.startswith(prompt):
                    #     chosen = chosen[len(prompt):]
                    # if rejected.startswith(prompt):
                    #     rejected = rejected[len(prompt):]
                    print(ds, 'prompt\n        ', fmt(prompt))
                    print(ds, 'chosen\n        ', fmt(chosen))
                    print(ds, 'rejected\n        ', fmt(rejected))
                    print()
    else:
        raise ValueError(f"Invalid section: {section}")
            

if __name__ == '__main__':
    main()
