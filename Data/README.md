## Rationale Generation

You can use the codes `chatgpt_abductive_reasoning.py` to generate the rationales. Remember to modify the path to different datasets and the OpenAI API key.

We also provide our generated rationales with ChatGPT3.5: `harmc_rationale.json`, `harmp_rationale.json`, and `FHM_rationale.json`.


## Image Data

The original image files can be found at [Harm-C](https://drive.google.com/file/d/1dxMrnyXcED-85HCcQiA_d5rr8acwl6lp/view?usp=sharing), [Harm-P](https://drive.google.com/file/d/1fw850yxKNqzpRpQKH88D13yfrwX1MLde/view?usp=sharing) and [FHM](https://hatefulmemeschallenge.com/#download).

## Data Preprocess

To separate the text and image in the memes, we first in-paint the memes by combining MMOCR (Kuang et al., 2021) with SAM (Kirillov et al., 2023) to extract the text and pure image. Then during the captioning process, since the focus of this work is primarily on the multimodal reasoning for harmful meme detection from a fresh perspective on harnessing LLMs, we apply a pre-trained image captioning model ClipCap (Mokady et al., 2021) used in recent work (Cao et al., 2022), to generate textual descriptions about the dominant objects or events in the memes’ image, which is utilized as the inputs into LLMs for abductive reasoning. To generate the rationale for each meme, we employed ChatGPT (Ouyang et al., 2022), a widely used LLM developed by OpenAI, specifically utilizing the “gpt-3.5-turbo” version. Drawing the practice of previous work like MaskPrompt (Cao et al., 2022) on FHM data preprocessing, the input text is augmented with image entities and demographic information in the FHM data preprocessing for a fair comparison with the baseline.
