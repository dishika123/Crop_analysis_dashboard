from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)
import streamlit as st

# Load environment variables from .env file
load_dotenv()



# read csv file
df_main = pd.read_csv("./data/merge.csv").fillna(0)
df_pattern = pd.read_csv("./data/crop_pattern.csv")
df = df_main.merge(df_pattern, on=["Crop", "Season"], how="left")

# Use a local Ollama model (ensure it's pulled: `ollama pull llama3.2`)
model = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434",
    temperature=0,
)

# Create a pandas agent with ReAct (works with Ollama)
agent = create_pandas_dataframe_agent(
    llm=model,
    df=df,
    agent_type="zero-shot-react-description",
    verbose=True,
    allow_dangerous_code=True,  # required for Python REPL tool
)

# then let's add some pre and sufix prompt
CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""

st.title("Database AI Agent with LangChain")

st.write("### Dataset Preview")
st.write(df.head())

# User input for the question
st.write("### Ask a Question")
question = st.text_input(
    "Enter your question about the dataset:",
    "Which grade has the highest average base salary, and compare the average female pay vs male pay?",
)

# Run the agent and display the result
if st.button("Run Query"):
    QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
    res = agent.invoke(QUERY)
    st.write("### Final Answer")
    st.markdown(res["output"])