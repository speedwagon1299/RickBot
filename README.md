# RickBotttttt

https://colab.research.google.com/drive/1uewENTh8UNexrHzrVXyDA8pQymrTm9p1?usp=sharing

Click on the colab link to run your version of the notebook.
Run all the cells and you can chat with Rick Sanchez's sentient DNA in the end of the file
The notebook is designed to be used on Colab so the dependencies are listed accordingly
To use unsloth ai, some packages like Triton are not present in Windows so a Google Colab environment is strongly suggested


*Method of training*

1. Found the Rick and Morty Scripts dataset and transformed it to a dialogue and 'n' context model data (Saved Under Input/R&Mdata.csv)
2. Used Unsloth.ai notebook and modified prompt format to finetune Llama-3 8B to respond like Rick Sanchez using attention to previous 'n'(= 6) dialogues to gain context
3. Fine tuned with best hyperparameter to ensure maximum clarity and prevent overtraining which usually lead to garbage repetitive response
4. Built a function to allow user to chat with the Bot with respect to past conversation history by maintaining previous dialogues as inputs to current dialogue for the LLM. 