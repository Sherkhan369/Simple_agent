from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
import chainlit as cl 
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

#stept 1
provider = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

#step2 
model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash",
    openai_client=  provider
)

run_config = RunConfig(
    model=model,
    model_provider= provider,
    tracing_disabled= True,
)
#step 3
agent1 = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. You answer questions and provide information based on the user's input.",
    
)
#step 4
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Welcome to the Gemini Assistant! How can I help you today?").send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    result = await Runner.run(
        agent1,
        input=history,
        run_config=run_config,
)
    history.append({"roll": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content= result.final_output).send()
    