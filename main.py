from dotenv import load_dotenv
import os
import gradio as gr
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are a study aide. Your whole purpose is to serve your user. Tend to their academic requests, and make sure they 
    have fully understood each and everything you are supposed to explain. Make no assumptions of prior knowledge. You 
    will assume that your user is learning everything for th efirst time. Every time you receive a problem, you will 
    start by asking the user how much they know about the fundaments of the problem at hand; regardless of whether or 
    not the fundamental keywords are present in the problem. You will always first gather relevant information about 
    the user's understanding of a concept before you proceed to tackle it. You will do this by diagnosing the user's 
    knowledge of relevant parameters. For example, if a user poses a problem about integration, you will identify the 
    techniques needed to solve the specific problem and then ask the user if they are familiar with these techniques.  
    Do this in less than six successive prompts. Do not provide an immediate explanation of each diagnostic parameter. 
    You will first gather information of what they know or do not know. If the user answers something that suggests they 
    are slightly familiar with the concept, assume they are totally unfamiliar with it. Once all five questions have 
    been asked, the diagnosis is complete. You will now begin exlaining each relevant parameter in order of how you 
    asked them; each in a separate response. In order for you to advance to successive parameters until finally 
    answering the question after the final parameter (in which you demonstrate the concepts you will have explained), 
    you will prompt the user to reply whether they are comfortable with your explanation or not. On solving a problem or 
    providing an answer, you will construct a similar trial practice question to test their knoweldge. 
    If necessary, bundle diagnostic parameters together. You may ask a minimum of one and a maximum of five inquiries 
    on the user's comfort with different diagnostic prompts.  
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0,
)

prompt=ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")
])

chat_chain = prompt | llm | StrOutputParser()
#or chat_chain = chain(prompt, llm, stroutputparser)

print("Hi I am Albert, how can I help you today?")

def chat(user_prompt,history): #this will use as input the values of the list in the (gradiotextboxvariable).submit method.
    print(user_prompt,history)
    langchain_history=[]
    for dialog in history: #the list of dictionaries; each with the roles and content as key value pairs, is
        # iterated over. The dictionary pairs alternate between those with a user role and those with an assistant role.
        if dialog ['role']=='user': #if the iteration is on a dictionary whose role is user,
            langchain_history.append(HumanMessage(content=dialog['content'])) #the content of this dictionary is
            # detected by key indexing (in the dictionary 'dialog' as defined by the for loop, with the key 'content').
            # This dictionary will provide the referenced value to a content parameter in the HumanMessage class constr.
        elif dialog['role']=='assistant':
            langchain_history.append(AIMessage(content=dialog['content']))

    start_time = time.time()
    response = chat_chain.invoke({"input": user_prompt, "history": langchain_history})
    stop_time = time.time()
    latency = stop_time - start_time

    print (f"Einstein took {latency:.2f} seconds to reply")

    return "", history + [{'role':'user', 'content':user_prompt},
                            {'role':'assistant', 'content':response}]

def clear_chat():
    return "", []

page = gr.Blocks(
    title="Chat with Einstein",
    theme=gr.themes.Soft()
)

with page:
    gr.Markdown (
        """
        # Chat with Einstein
        Welcome to your personal conversation with Albert Einstein!
        """
    )
    chatbot = gr.Chatbot(type='messages',
                        avatar_images=[None, 'einstein.png'],
                        show_label=False)
    msg = gr.Textbox(show_label=False, placeholder="Ask Einstein Anything")
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear = gr.Button("Clear Chat")
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch(share=True)
    # history.append({"role": "user", "content": user_prompt})
    # response=llm.invoke([
    #     {"role":"system", "content":system_prompt}+history])
    # print (f"Albert: {response.content}")

# while True:
#     user_input = input("You: ")
#
#     if user_input == "exit":
#         break
#     print(f"Cool, thanks for sharing that {user_input}")
#
# response = "Hi I am Albert, how can I help you today?"
# print(response)
