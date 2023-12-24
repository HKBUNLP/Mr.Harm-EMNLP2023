import json
import time
import openai
import pickle
openai.api_key = "XXXXX"

data = dict()
id = 0
with open("./FHM/mem_train.json", "r", encoding='utf8') as fin:
    data_list = json.load(fin)
    # for item in jsonlines.Reader(fin):
    #     data[id]=item
    #     id += 1
    fin.close()
for data_item in data_list:
    data[id] = data_item
    id += 1

with open('./FHM/clean_captions.pkl','rb') as f:
    caption_dict = pickle.load(f)
    f.close()

cids = list(data.keys())
pred = {}


system_prompt = "You have been specially designed to perform abductive reasoning for the harmful meme detection task. Your primary function is that, according to a Harmfulness label about an Image with a text embedded, " \
                "please provide me a streamlined rationale, without explicitly indicating the label, why it is classified as the given Harmfulness label. " \
                "The image and the textual content in the meme are often uncorrelated, but its overall semantics is presented holistically. Thus it is important to note that you are prohibited from relying on your own imagination, as your goal is to provide the most accurate and reliable rationale possible " \
                "so that people can infer the harmfulness according to your reasoning about the background context and relationship between the given text and image caption."
count = 0
while count < len(cids):

    cid = cids[count]
    # if data[cid]["id"] in pred:
    #     count+=1
    #     print(count)
    #     continue
    try:
        text = data[cid]["clean_sent"].replace('\n', ' ').strip('\n')
        caption = caption_dict[data[cid]["img"].strip('.png')].strip('\n')
        if data[cid]["label"] == 1:
            label = 'harmful'
        else:
            label = 'harmless'

        user_prompt = f"Given a Text: '{text}', which is embedded in an Image: '{caption}'; and a harmfulness label '{label}', please give me a streamlined rationale associated with the meme, without explicitly indicating the label, why it is reasoned as {label}."

        reply = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=256,
        )
        ans = reply["choices"][0]["message"]["content"]
        print(user_prompt)
        print(ans)
        print(count)

        pred[data[cid]["img"].strip('.png')] = ans.lower()
        with open("clipcap_FHM_rationale_.pkl", "wb") as fout:
            pickle.dump(pred, fout)
            fout.close()
        count += 1
    except:
        print("Let's have a sleep.")
        time.sleep(61)